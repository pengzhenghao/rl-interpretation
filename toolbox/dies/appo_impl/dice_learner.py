"""Helper class for AsyncSamplesOptimizer."""

import logging
import math
import threading
from collections import defaultdict

import numpy as np
from ray.rllib.evaluation.metrics import LEARNER_STATS_KEY, get_learner_stats
# from ray.rllib.optimizers.aso_minibatch_buffer import MinibatchBuffer
from ray.rllib.optimizers.multi_gpu_impl import LocalSyncParallelOptimizer
from ray.rllib.policy.sample_batch import SampleBatch, DEFAULT_POLICY_ID, \
    MultiAgentBatch
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.timer import TimerStat
from ray.rllib.utils.window_stat import WindowStat
from six.moves import queue


def _averaged(kv):
    out = {}
    for k, v in kv.items():
        if v[0] is not None and not isinstance(v[0], dict):
            out[k] = np.mean(v)
    return out


tf = try_import_tf()

logger = logging.getLogger(__file__)


class AsyncLearnerThread(threading.Thread):
    """Background thread that updates the local model from sample trajectories.

    This is for use with AsyncSamplesOptimizer.

    The learner thread communicates with the main thread through Queues. This
    is needed since Ray operations can only be run on the main thread. In
    addition, moving heavyweight gradient ops session runs off the main thread
    improves overall throughput.
    """

    def __init__(self, local_worker, minibatch_buffer_size, num_sgd_iter,
                 learner_queue_size, learner_queue_timeout):
        """Initialize the learner thread.

        Arguments:
            local_worker (RolloutWorker): process local rollout worker holding
                policies this thread will call learn_on_batch() on
            minibatch_buffer_size (int): max number of train batches to store
                in the minibatching buffer
            num_sgd_iter (int): number of passes to learn on per train batch
            learner_queue_size (int): max size of queue of inbound
                train batches to this thread
            learner_queue_timeout (int): raise an exception if the queue has
                been empty for this long in seconds
        """
        threading.Thread.__init__(self)
        self.learner_queue_size = WindowStat("size", 50)
        self.local_worker = local_worker
        self.inqueue = queue.Queue(maxsize=learner_queue_size)
        self.outqueue = queue.Queue()
        self.minibatch_buffer = MinibatchBuffer(
            inqueue=self.inqueue,
            size=minibatch_buffer_size,
            timeout=learner_queue_timeout,
            num_sgd_iter=num_sgd_iter,
            init_num_passes=num_sgd_iter)

        self.queue_timer = TimerStat()
        self.grad_timer = TimerStat()
        self.load_timer = TimerStat()
        self.load_wait_timer = TimerStat()
        self.daemon = True
        self.weights_updated = False
        self.stats = {"train_timesteps": 0}
        self.stopped = False
        self.num_steps = 0
        self.num_sgd_iter = num_sgd_iter

    def run(self):
        while not self.stopped:
            self.step()

    def step(self):
        with self.queue_timer:
            batch, _ = self.minibatch_buffer.get()
        with self.grad_timer:
            fetches = self.local_worker.learn_on_batch(batch)
            self.weights_updated = True
            self.stats.update(get_learner_stats(fetches))
            self.stats["train_timesteps"] += batch.count
            self.num_steps += 1
            self.stats["update_steps"] = self.num_steps
        self.outqueue.put(batch.count)
        self.learner_queue_size.push(self.inqueue.qsize())


class SyncLearnerThread(threading.Thread):
    """Background thread that updates the local model from sample trajectories.

    PZH: We wish to mock the behavior of a normal PPO pipeline. This mean that,
    allowing multiple SGD epochs and mini-batching within each SGD epochs.
    These requirements are not supported by current APPO implementation.

    The learner thread communicates with the main thread through Queues. This
    is needed since Ray operations can only be run on the main thread. In
    addition, moving heavyweight gradient ops session runs off the main thread
    improves overall throughput.
    """

    def __init__(self, local_worker, minibatch_buffer_size, num_sgd_iter,
                 learner_queue_size, learner_queue_timeout, num_gpus,
                 sgd_batch_size):
        threading.Thread.__init__(self)
        self.learner_queue_size = WindowStat("size", 50)
        self.local_worker = local_worker
        self.inqueue = queue.Queue(maxsize=learner_queue_size)
        self.outqueue = queue.Queue()

        self.minibatch_buffer = MinibatchBuffer(
            inqueue=self.inqueue,
            size=1,
            # size=minibatch_buffer_size,
            timeout=learner_queue_timeout,
            # num_sgd_iter=num_sgd_iter,
            num_sgd_iter=1,
            init_num_passes=1)

        self.queue_timer = TimerStat()
        self.grad_timer = TimerStat()
        self.load_timer = TimerStat()
        self.load_wait_timer = TimerStat()
        self.daemon = True
        self.weights_updated = False
        self.stats = {"train_timesteps": 0}
        self.stopped = False
        self.num_steps = 0
        self.num_sgd_iter = num_sgd_iter

        # ===== Copied the initialization in multi_gpu_optimizer
        if not num_gpus:
            self.devices = ["/cpu:0"]
        else:
            self.devices = [
                "/gpu:{}".format(i) for i in range(int(math.ceil(num_gpus)))
            ]

        self.batch_size = int(sgd_batch_size / len(self.devices)) * len(
            self.devices)
        assert self.batch_size % len(self.devices) == 0
        assert self.batch_size >= len(self.devices), "batch size too small"
        self.per_device_batch_size = int(self.batch_size / len(self.devices))

        self.policies = dict(
            local_worker.foreach_trainable_policy(lambda p, i: (i, p)))

        self.optimizers = {}
        with local_worker.tf_sess.graph.as_default():
            with local_worker.tf_sess.as_default():
                for policy_id, policy in self.policies.items():
                    with tf.variable_scope(policy_id, reuse=tf.AUTO_REUSE):
                        if policy._state_inputs:
                            rnn_inputs = policy._state_inputs + [
                                policy._seq_lens
                            ]
                        else:
                            rnn_inputs = []
                        self.optimizers[policy_id] = (
                            LocalSyncParallelOptimizer(
                                policy._optimizer, self.devices,
                                [v
                                 for _, v in policy._loss_inputs], rnn_inputs,
                                self.per_device_batch_size, policy.copy))
                self.sess = local_worker.tf_sess
                self.sess.run(tf.global_variables_initializer())

    def run(self):
        while not self.stopped:
            self.step()

    def step(self):
        with self.queue_timer:
            batch, _ = self.minibatch_buffer.get()
            # Handle everything as if multiagent
            if isinstance(batch, SampleBatch):
                batch = MultiAgentBatch({
                    DEFAULT_POLICY_ID: batch
                }, batch.count)

            # TODO maybe we should do the normalization here

        num_loaded_tuples = {}
        with self.load_timer:
            for policy_id, batch in batch.policy_batches.items():
                if policy_id not in self.policies:
                    continue

                policy = self.policies[policy_id]
                policy._debug_vars()
                tuples = policy._get_loss_inputs_dict(
                    batch, shuffle=True)
                data_keys = [ph for _, ph in policy._loss_inputs]
                if policy._state_inputs:
                    state_keys = policy._state_inputs + [policy._seq_lens]
                else:
                    state_keys = []
                num_loaded_tuples[policy_id] = (
                    self.optimizers[policy_id].load_data(
                        self.sess, [tuples[k] for k in data_keys],
                        [tuples[k] for k in state_keys]))

        fetches = {}
        with self.grad_timer:
            for policy_id, tuples_per_device in num_loaded_tuples.items():
                optimizer = self.optimizers[policy_id]
                num_batches = max(
                    1,
                    int(tuples_per_device) // int(self.per_device_batch_size))
                logger.debug("== sgd epochs for {} ==".format(policy_id))
                for i in range(self.num_sgd_iter):
                    iter_extra_fetches = defaultdict(list)
                    permutation = np.random.permutation(num_batches)
                    for batch_index in range(num_batches):
                        batch_fetches = optimizer.optimize(
                            self.sess, permutation[batch_index] *
                                       self.per_device_batch_size)
                        for k, v in batch_fetches[LEARNER_STATS_KEY].items():
                            iter_extra_fetches[k].append(v)
                    logger.debug(
                        "{} {}".format(i, _averaged(iter_extra_fetches)))
                fetches[policy_id] = _averaged(iter_extra_fetches)

        # Not support multiagent recording now.
        self.stats.update(fetches["default_policy"])
        self.stats["train_timesteps"] += tuples_per_device
        self.num_steps += 1
        self.stats["update_steps"] = self.num_steps

        self.outqueue.put(batch.count)
        self.learner_queue_size.push(self.inqueue.qsize())
        self.weights_updated = True

        if self.minibatch_buffer.is_empty():
            # Send signal to optimizer
            self.outqueue.put(None)


class MinibatchBuffer:
    """Ring buffer of recent data batches for minibatch SGD.

    This is for use with AsyncSamplesOptimizer.

    Copied from ray/rllib/optimizers/aso_minibatch_buffer.py
    Rewrite to allow mini batching in one SGD epoch
    """

    def __init__(self, inqueue, size, timeout, num_sgd_iter, init_num_passes=1):
        """Initialize a minibatch buffer.

        Arguments:
           inqueue: Queue to populate the internal ring buffer from.
           size: Max number of data items to buffer.
           timeout: Queue timeout
           num_sgd_iter: Max num times each data item should be emitted.
           init_num_passes: Initial max passes for each data item
       """
        self.inqueue = inqueue
        self.size = size
        self.timeout = timeout
        self.max_ttl = num_sgd_iter
        self.cur_max_ttl = init_num_passes
        self.buffers = [None] * size
        self.ttl = [0] * size
        self.idx = 0

    def get(self):
        """Get a new batch from the internal ring buffer.

        Returns:
           buf: Data item saved from inqueue.
           released: True if the item is now removed from the ring buffer.
        """
        if self.ttl[self.idx] <= 0:
            self.buffers[self.idx] = self.inqueue.get(timeout=self.timeout)
            self.ttl[self.idx] = self.cur_max_ttl
            if self.cur_max_ttl < self.max_ttl:
                self.cur_max_ttl += 1
        buf = self.buffers[self.idx]
        self.ttl[self.idx] -= 1
        released = self.ttl[self.idx] <= 0
        if released:
            self.buffers[self.idx] = None
        self.idx = (self.idx + 1) % len(self.buffers)
        return buf, released

    def is_empty(self):
        return all(d is None for d in self.buffers)
