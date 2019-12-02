"""Copied from rllib. The main modification is that we allow the
cross-policies data processing. Besides, we also modify a bit to fit the
changes in multi_gpu_impl.py."""
from __future__ import absolute_import, division, print_function

import logging
import math
from collections import defaultdict

import numpy as np
import ray
from ray.rllib.evaluation.metrics import LEARNER_STATS_KEY
from ray.rllib.optimizers import LocalMultiGPUOptimizer
from ray.rllib.optimizers.policy_optimizer import PolicyOptimizer
from ray.rllib.optimizers.rollout import collect_samples
from ray.rllib.policy.sample_batch import SampleBatch, DEFAULT_POLICY_ID, \
    MultiAgentBatch
from ray.rllib.policy.tf_policy import TFPolicy
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.timer import TimerStat

from toolbox.modified_rllib.multi_gpu_impl import \
    LocalSyncParallelOptimizerModified

tf = try_import_tf()

logger = logging.getLogger(__name__)


class LocalMultiGPUOptimizerModified(LocalMultiGPUOptimizer):
    def __init__(
            self,
            workers,
            no_split_list,
            process_multiagent_batch_fn=None,
            sgd_batch_size=128,
            num_sgd_iter=10,
            sample_batch_size=200,
            num_envs_per_worker=1,
            train_batch_size=1024,
            num_gpus=0,
            standardize_fields=[],
            shuffle_sequences=True
    ):
        """We introduce a new feature which the original rllib do not have,
        that is allow to parse the cross policy information and store then
        in an object, which will be passed to each policy."""
        self.process_multiagent_batch_fn = process_multiagent_batch_fn
        PolicyOptimizer.__init__(self, workers)

        self.batch_size = sgd_batch_size
        self.num_sgd_iter = num_sgd_iter
        self.num_envs_per_worker = num_envs_per_worker
        self.sample_batch_size = sample_batch_size
        self.train_batch_size = train_batch_size
        self.shuffle_sequences = shuffle_sequences
        if not num_gpus:
            self.devices = ["/cpu:0"]
        else:
            self.devices = [
                "/gpu:{}".format(i) for i in range(int(math.ceil(num_gpus)))
            ]
        self.batch_size = int(sgd_batch_size / len(self.devices)
                              ) * len(self.devices)
        assert self.batch_size % len(self.devices) == 0
        assert self.batch_size >= len(self.devices), "batch size too small"
        self.per_device_batch_size = int(self.batch_size / len(self.devices))
        self.sample_timer = TimerStat()
        self.load_timer = TimerStat()
        self.grad_timer = TimerStat()
        self.update_weights_timer = TimerStat()
        self.standardize_fields = standardize_fields

        logger.info("LocalMultiGPUOptimizer devices {}".format(self.devices))

        self.policies = dict(
            self.workers.local_worker().
                foreach_trainable_policy(lambda p, i: (i, p))
        )
        logger.debug("Policies to train: {}".format(self.policies))
        for policy_id, policy in self.policies.items():
            if not isinstance(policy, TFPolicy):
                raise ValueError(
                    "Only TF graph policies are supported with multi-GPU. "
                    "Try setting `simple_optimizer=True` instead."
                )

        # per-GPU graph copies created below must share vars with the policy
        # reuse is set to AUTO_REUSE because Adam nodes are created after
        # all of the device copies are created.
        self.optimizers = {}
        with self.workers.local_worker().tf_sess.graph.as_default():
            with self.workers.local_worker().tf_sess.as_default():
                for policy_id, policy in self.policies.items():
                    with tf.variable_scope(policy_id, reuse=tf.AUTO_REUSE):
                        if policy._state_inputs:
                            rnn_inputs = policy._state_inputs + [
                                policy._seq_lens
                            ]
                        else:
                            rnn_inputs = []
                        self.optimizers[policy_id] = (
                            LocalSyncParallelOptimizerModified(
                                policy._optimizer, self.devices, {
                                    (i, k): v
                                    for i,
                                        (k, v) in
                                    enumerate(policy._loss_inputs)
                                    if k not in no_split_list
                                }, {
                                    (i, k): v
                                    for i,
                                        (k, v) in
                                    enumerate(policy._loss_inputs)
                                    if k in no_split_list
                                }, [
                                    (i, k) for i,
                                               (k, _) in
                                    enumerate(policy._loss_inputs)
                                ], rnn_inputs, self.per_device_batch_size,
                                policy.copy
                            )
                        )

                self.sess = self.workers.local_worker().tf_sess
                self.sess.run(tf.global_variables_initializer())

    def step(self):
        """Override the original codes to add other policies infos."""
        with self.update_weights_timer:
            if self.workers.remote_workers():
                weights = ray.put(self.workers.local_worker().get_weights())
                for e in self.workers.remote_workers():
                    e.set_weights.remote(weights)

        with self.sample_timer:
            if self.workers.remote_workers():
                samples = collect_samples(
                    self.workers.remote_workers(), self.sample_batch_size,
                    self.num_envs_per_worker, self.train_batch_size
                )
                if samples.count > self.train_batch_size * 2:
                    logger.info(
                        "Collected more training samples than expected "
                        "(actual={}, train_batch_size={}). ".
                        format(samples.count, self.train_batch_size) +
                        "This may be because you have many workers or "
                        "long episodes in 'complete_episodes' batch mode."
                    )
            else:
                samples = []
                while sum(s.count for s in samples) < self.train_batch_size:
                    samples.append(self.workers.local_worker().sample())
                samples = SampleBatch.concat_samples(samples)

            # Handle everything as if multiagent
            if isinstance(samples, SampleBatch):
                samples = MultiAgentBatch(
                    {DEFAULT_POLICY_ID: samples}, samples.count
                )

        for policy_id, policy in self.policies.items():
            if policy_id not in samples.policy_batches:
                continue

            batch = samples.policy_batches[policy_id]
            for field in self.standardize_fields:
                value = batch[field]
                standardized = (value - value.mean()) / max(1e-4, value.std())
                batch[field] = standardized

        num_loaded_tuples = {}

        if self.process_multiagent_batch_fn is not None:
            cross_policy_obj = self.process_multiagent_batch_fn(samples, self)
        else:
            cross_policy_obj = None

        with self.load_timer:
            for policy_id, batch in samples.policy_batches.items():
                if policy_id not in self.policies:
                    continue

                policy = self.policies[policy_id]
                policy._debug_vars()
                tuples = policy._get_loss_inputs_dict(
                    batch,
                    shuffle=self.shuffle_sequences,
                    cross_policy_obj=cross_policy_obj,
                    policy_id=policy_id
                )  ### HERE!
                data_keys = [ph for _, ph in policy._loss_inputs]
                if policy._state_inputs:
                    state_keys = policy._state_inputs + [policy._seq_lens]
                else:
                    state_keys = []
                num_loaded_tuples[policy_id] = (
                    self.optimizers[policy_id].load_data(
                        self.sess, [tuples[k] for k in data_keys],
                        [tuples[k] for k in state_keys]
                    )
                )

        fetches = {}
        with self.grad_timer:
            for policy_id, tuples_per_device in num_loaded_tuples.items():
                optimizer = self.optimizers[policy_id]
                num_batches = max(
                    1,
                    int(tuples_per_device) // int(self.per_device_batch_size)
                )
                logger.debug("== sgd epochs for {} ==".format(policy_id))
                for i in range(self.num_sgd_iter):
                    iter_extra_fetches = defaultdict(list)
                    permutation = np.random.permutation(num_batches)
                    for batch_index in range(num_batches):
                        batch_fetches = optimizer.optimize(
                            self.sess, permutation[batch_index] *
                                       self.per_device_batch_size
                        )
                        for k, v in batch_fetches[LEARNER_STATS_KEY].items():
                            iter_extra_fetches[k].append(v)
                    logger.debug(
                        "{} {}".format(i, _averaged(iter_extra_fetches))
                    )
                fetches[policy_id] = _averaged(iter_extra_fetches)

        self.num_steps_sampled += samples.count
        self.num_steps_trained += tuples_per_device * len(self.devices)
        self.learner_stats = fetches
        return fetches


class LocalMultiGPUOptimizerCorrectedNumberOfSampled(LocalMultiGPUOptimizer):
    def step(self):
        with self.update_weights_timer:
            if self.workers.remote_workers():
                weights = ray.put(self.workers.local_worker().get_weights())
                for e in self.workers.remote_workers():
                    e.set_weights.remote(weights)

        with self.sample_timer:
            if self.workers.remote_workers():
                samples = collect_samples(
                    self.workers.remote_workers(), self.sample_batch_size,
                    self.num_envs_per_worker, self.train_batch_size
                )
                if samples.count > self.train_batch_size * 2:
                    logger.info(
                        "Collected more training samples than expected "
                        "(actual={}, train_batch_size={}). ".
                        format(samples.count, self.train_batch_size) +
                        "This may be because you have many workers or "
                        "long episodes in 'complete_episodes' batch mode."
                    )
            else:
                samples = []
                while sum(s.count for s in samples) < self.train_batch_size:
                    samples.append(self.workers.local_worker().sample())
                samples = SampleBatch.concat_samples(samples)

            # Handle everything as if multiagent
            if isinstance(samples, SampleBatch):
                samples = MultiAgentBatch(
                    {DEFAULT_POLICY_ID: samples}, samples.count
                )

        for policy_id, policy in self.policies.items():
            if policy_id not in samples.policy_batches:
                continue

            batch = samples.policy_batches[policy_id]
            for field in self.standardize_fields:
                value = batch[field]
                standardized = (value - value.mean()) / max(1e-4, value.std())
                batch[field] = standardized

        num_loaded_tuples = {}
        with self.load_timer:
            for policy_id, batch in samples.policy_batches.items():
                if policy_id not in self.policies:
                    continue

                policy = self.policies[policy_id]
                policy._debug_vars()
                tuples = policy._get_loss_inputs_dict(
                    batch, shuffle=self.shuffle_sequences
                )
                data_keys = [ph for _, ph in policy._loss_inputs]
                if policy._state_inputs:
                    state_keys = policy._state_inputs + [policy._seq_lens]
                else:
                    state_keys = []
                num_loaded_tuples[policy_id] = (
                    self.optimizers[policy_id].load_data(
                        self.sess, [tuples[k] for k in data_keys],
                        [tuples[k] for k in state_keys]
                    )
                )

        fetches = {}
        with self.grad_timer:
            for policy_id, tuples_per_device in num_loaded_tuples.items():
                optimizer = self.optimizers[policy_id]
                num_batches = max(
                    1,
                    int(tuples_per_device) // int(self.per_device_batch_size)
                )
                logger.debug("== sgd epochs for {} ==".format(policy_id))
                for i in range(self.num_sgd_iter):
                    iter_extra_fetches = defaultdict(list)
                    permutation = np.random.permutation(num_batches)
                    for batch_index in range(num_batches):
                        batch_fetches = optimizer.optimize(
                            self.sess, permutation[batch_index] *
                                       self.per_device_batch_size
                        )
                        for k, v in batch_fetches[LEARNER_STATS_KEY].items():
                            iter_extra_fetches[k].append(v)
                    logger.debug(
                        "{} {}".format(i, _averaged(iter_extra_fetches))
                    )
                fetches[policy_id] = _averaged(iter_extra_fetches)
        self.num_steps_sampled += np.mean(
            [b.count for b in samples.policy_batches.values()], dtype=np.int64
        )
        self.num_steps_trained += np.mean(
            list(num_loaded_tuples.values()), dtype=np.int64
        ) * len(self.devices)
        self.learner_stats = fetches
        return fetches


def _averaged(kv):
    out = {}
    for k, v in kv.items():
        if v[0] is not None and not isinstance(v[0], dict):
            out[k] = np.mean(v)
    return out
