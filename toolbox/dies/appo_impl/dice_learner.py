"""Helper class for AsyncSamplesOptimizer."""

import logging
import threading
import time

from ray.rllib.evaluation.metrics import get_learner_stats
# from ray.rllib.optimizers.aso_minibatch_buffer import MinibatchBuffer
from ray.rllib.utils.timer import TimerStat
from ray.rllib.utils.window_stat import WindowStat
from six.moves import queue

logger = logging.getLogger(__file__)


class LearnerThread(threading.Thread):
    """Background thread that updates the local model from sample trajectories.

    This is for use with AsyncSamplesOptimizer.

    The learner thread communicates with the main thread through Queues. This
    is needed since Ray operations can only be run on the main thread. In
    addition, moving heavyweight gradient ops session runs off the main thread
    improves overall throughput.
    """

    def __init__(self, local_worker, sgd_minibatch_size, num_sgd_iter,
                 learner_queue_size, learner_queue_timeout, train_batch_size=1,
                 sync_sampling=False):
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

        # self.num_batches = max(
        #     1, int(train_batch_size) // int(sgd_minibatch_size))

        if not sync_sampling:
            self.minibatch_buffer = MinibatchBuffer(
                inqueue=self.inqueue,
                size=1,  # TODO this is set to 1
                timeout=learner_queue_timeout,
                num_sgd_iter=num_sgd_iter,
                init_num_passes=num_sgd_iter)
        else:
            # TODO change the name of this buffer
            self.minibatch_buffer = MinibatchBufferNew(
                inqueue=self.inqueue,
                train_batch_size=train_batch_size,
                timeout=learner_queue_timeout,
                num_sgd_iter=num_sgd_iter,
                shuffle=True,  # TODO set a flog for this
                mini_batch_size=sgd_minibatch_size
            )

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
        self.sync_sampling = sync_sampling

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

        # self.outqueue.put(batch.count)

        # Since all batch are repeated num_sgd_iter times, so as a workaround,
        # we divide the reported trained steps by num_sgd_iter
        self.outqueue.put(batch.count / self.num_sgd_iter)
        self.learner_queue_size.push(self.inqueue.qsize())


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


class MinibatchBufferNew:
    """Ring buffer of recent data batches for minibatch SGD.

    This is for use with AsyncSamplesOptimizer.

    Copied from ray/rllib/optimizers/aso_minibatch_buffer.py
    Rewrite to allow mini batching in one SGD epoch
    """

    def __init__(self, inqueue, train_batch_size, timeout, num_sgd_iter,
                 # init_num_passes=1,
                 shuffle=False, mini_batch_size=1):
        """Initialize a minibatch buffer.

        Arguments:
           inqueue: Queue to populate the internal ring buffer from.
           size: Max number of data items to buffer.
           timeout: Queue timeout
           num_sgd_iter: Max num times each data item should be emitted.
           init_num_passes: Initial max passes for each data item
       """
        self.inqueue = inqueue

        self.mini_batch_size = mini_batch_size
        self.train_batch_size = train_batch_size
        self.num_minibatch = max(
            1, int(train_batch_size) // int(mini_batch_size))

        print("***** Number of minibatch: {} *****".format(self.num_minibatch))
        print("***** Number of sgd iter: {} *****".format(num_sgd_iter))

        self.timeout = timeout
        self.max_ttl = num_sgd_iter
        # self.cur_max_ttl = init_num_passes
        self.buffers = [None] * self.num_minibatch
        self.ttl = [0] * self.num_minibatch
        self.idx = 0
        self.shuffle = shuffle

        self._debug_count = 0
        self._debug_count_batch = 0
        self._debug_time = time.time()

    def fill(self):
        """Split the train batch into mini batches"""

        # print("***** ===== Current {} ===== *****".format(self._debug_count))

        self._debug_count_batch += 1

        # Get data
        train_batch = self.inqueue.get(timeout=self.timeout)
        print("***** Receive {} batches from input queue. after getting {} "
              "minibatches. Used time {}. This batch size {}.".format(
            self._debug_count_batch, self._debug_count,
            time.time() - self._debug_time, train_batch.count))
        self._debug_count = 0
        self._debug_time = time.time()

        # Shuffle is necessary
        if self.shuffle:
            train_batch.shuffle()

        # Split
        assert train_batch.count == self.train_batch_size

        for i in range(self.num_minibatch):
            self.buffers[i] = train_batch.slice(i * self.mini_batch_size,
                                                (i + 1) * self.mini_batch_size)
            self.ttl[i] = self.max_ttl

        self.idx = 0

    def is_empty(self):
        return all(d is None for d in self.buffers)

    def get(self):
        """Get a new batch from the internal ring buffer.

        Returns:
           buf: Data item saved from inqueue.
           released: True if the item is now removed from the ring buffer.
        """
        # if self.ttl[self.idx] <= 0:
        # self.buffers[self.idx] = self.inqueue.get(timeout=self.timeout)
        # self.ttl[self.idx] = self.cur_max_ttl
        # if self.cur_max_ttl < self.max_ttl:
        #     self.cur_max_ttl += 1

        self._debug_count += 1

        if self.is_empty():
            self.fill()

        buf = self.buffers[self.idx]
        self.ttl[self.idx] -= 1

        released = self.ttl[self.idx] <= 0
        if released:
            self.buffers[self.idx] = None

        self.idx = (self.idx + 1) % len(self.buffers)
        return buf, released
