import collections
import logging
import sys

import numpy as np
import ray
from ray.rllib.optimizers.policy_optimizer import PolicyOptimizer
from ray.rllib.optimizers.replay_buffer import PrioritizedReplayBuffer, \
    ReplayBuffer
from ray.rllib.optimizers.sync_replay_optimizer import SyncReplayOptimizer
from ray.rllib.policy.sample_batch import SampleBatch, DEFAULT_POLICY_ID, \
    MultiAgentBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.compression import pack_if_needed, unpack_if_needed
from ray.rllib.utils.memory import ray_get_and_free

logger = logging.getLogger(__name__)


class ReplayBufferModified(ReplayBuffer):
    def add(self, obs_t, action, reward, obs_tp1, done, *others):
        """add action_logp"""
        data = (obs_t, action, reward, obs_tp1, done, *others)
        self._num_added += 1

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
            self._est_size_bytes += sum(sys.getsizeof(d) for d in data)
        else:
            self._storage[self._next_idx] = data
        if self._next_idx + 1 >= self._maxsize:
            self._eviction_started = True
        self._next_idx = (self._next_idx + 1) % self._maxsize
        if self._eviction_started:
            self._evicted_hit_stats.push(self._hit_count[self._next_idx])
            self._hit_count[self._next_idx] = 0

    def _encode_sample(self, idxes):
        """Add action_logps"""
        obses_t, actions, rewards, obses_tp1, dones = \
            [], [], [], [], []

        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, *_ = data
            obses_t.append(np.array(unpack_if_needed(obs_t), copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(unpack_if_needed(obs_tp1), copy=False))
            dones.append(done)
            # action_logps.append(logp)
            self._hit_count[i] += 1

        ret = [
            np.array(obses_t),
            np.array(actions),
            np.array(rewards),
            np.array(obses_tp1),
            np.array(dones)
        ]

        # Add other necessary information
        for item_id in range(5, len(data)):
            ret.append(np.array([self._storage[i][item_id] for i in idxes]))

        return ret


class SyncReplayOptimizerModified(SyncReplayOptimizer):
    def __init__(self, *args, **kwargs):
        super(SyncReplayOptimizerModified, self).__init__(*args, **kwargs)

        def new_buffer():
            return ReplayBufferModified(kwargs["buffer_size"])

        self.replay_buffers = collections.defaultdict(new_buffer)

    @override(PolicyOptimizer)
    def step(self):
        with self.update_weights_timer:
            if self.workers.remote_workers():
                weights = ray.put(self.workers.local_worker().get_weights())
                for e in self.workers.remote_workers():
                    e.set_weights.remote(weights)

        with self.sample_timer:
            if self.workers.remote_workers():
                batch = SampleBatch.concat_samples(
                    ray_get_and_free(
                        [
                            e.sample.remote()
                            for e in self.workers.remote_workers()
                        ]
                    )
                )
            else:
                batch = self.workers.local_worker().sample()

            # Handle everything as if multiagent
            if isinstance(batch, SampleBatch):
                batch = MultiAgentBatch(
                    {DEFAULT_POLICY_ID: batch}, batch.count
                )

            for policy_id, s in batch.policy_batches.items():
                for row in s.rows():
                    self.replay_buffers[policy_id].add(
                        pack_if_needed(row["obs"]),
                        row["actions"],
                        row["rewards"],
                        pack_if_needed(row["new_obs"]),
                        row["dones"],
                        None,
                        row["action_logp"],
                        # row["diversity_advantages"],
                        row["diversity_rewards"],
                        # row["diversity_value_targets"],
                        # row["my_logits"],
                        row["prev_actions"],
                        row["prev_rewards"]
                    )

        if self.num_steps_sampled >= self.replay_starts:
            self._optimize()

        self.num_steps_sampled += batch.count

    def _replay(self):
        samples = {}
        idxes = None
        with self.replay_timer:
            for policy_id, replay_buffer in self.replay_buffers.items():
                if self.synchronize_sampling:
                    if idxes is None:
                        idxes = replay_buffer.sample_idxes(
                            self.train_batch_size
                        )
                else:
                    idxes = replay_buffer.sample_idxes(self.train_batch_size)

                if isinstance(replay_buffer, PrioritizedReplayBuffer):
                    raise ValueError()
                    # (obses_t, actions, rewards, obses_tp1, dones, weights,
                    #  batch_indexes) = replay_buffer.sample_with_idxes(
                    #     idxes,
                    #     beta=self.prioritized_replay_beta.value(
                    #         self.num_steps_trained))
                else:
                    (
                        obses_t,
                        actions,
                        rewards,
                        obses_tp1,
                        dones,
                        _,
                        action_logp,
                        # diversity_advantages,
                        diversity_rewards,
                        # diversity_value_targets,
                        # my_logits,
                        prev_actions,
                        prev_rewards
                    ) = replay_buffer.sample_with_idxes(idxes)
                    weights = np.ones_like(rewards)
                    batch_indexes = -np.ones_like(rewards)
                samples[policy_id] = SampleBatch(
                    {
                        "obs": obses_t,
                        "actions": actions,
                        "rewards": rewards,
                        "new_obs": obses_tp1,
                        "dones": dones,
                        "weights": weights,
                        "batch_indexes": batch_indexes,
                        "action_logp": action_logp,
                        # "diversity_advantages": diversity_advantages,
                        "diversity_rewards": diversity_rewards,
                        # "diversity_value_targets": diversity_value_targets,
                        # "my_logits": my_logits,
                        "prev_actions": prev_actions,
                        "prev_rewards": prev_rewards
                    }
                )
        return MultiAgentBatch(samples, self.train_batch_size)
