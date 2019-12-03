import ray
from ray.rllib.agents.ddpg.ddpg_policy import DDPGTFPolicy, SampleBatch
from ray.rllib.agents.ddpg.td3 import TD3Trainer, TD3_DEFAULT_CONFIG
from ray.rllib.optimizers.sync_replay_optimizer import SyncReplayOptimizer, \
    PrioritizedReplayBuffer, MultiAgentBatch, ray_get_and_free, \
    pack_if_needed, \
    DEFAULT_POLICY_ID, get_learner_stats, np
from ray.tune.util import merge_dicts

DISABLE = "disable"
SHARE_SAMPLE = "share_sample"

cetd3_default_config = merge_dicts(
    TD3_DEFAULT_CONFIG,
    dict(mode=SHARE_SAMPLE)
    # dict(learn_with_peers=True, use_joint_dataset=False, mode=REPLAY_VALUES)
)


class SyncReplayOptimizerWithCooperativeExploration(SyncReplayOptimizer):
    def _replay(self):
        samples = super()._replay()

        # Add other's batch here.
        config = self.workers._local_config
        if config["mode"] == SHARE_SAMPLE:
            share_sample = SampleBatch.concat_samples(
                [batch for batch in samples.policy_batches.values()]
            )
            for pid in samples.policy_batches.keys():
                samples.policy_batches[pid] = share_sample
            samples.count = share_sample.count
        return samples

    def step(self):
        """We correct the number of sampled steps."""
        # The below codes are copied from Ray's SyncReplayOptimizer
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
                        weight=None
                    )
        if self.num_steps_sampled >= self.replay_starts:
            self._optimize()

        # Here!
        # self.num_steps_sampled += batch.count
        self.num_steps_sampled += np.mean(
            [b.count for b in batch.policy_batches.values()], dtype=np.int64
        )

    def _optimize(self):
        """We correct the number of trained agents."""
        # The below codes are copied from Ray's SyncReplayOptimizer
        samples = self._replay()
        with self.grad_timer:
            if self.before_learn_on_batch:
                samples = self.before_learn_on_batch(
                    samples,
                    self.workers.local_worker().policy_map,
                    self.train_batch_size
                )
            info_dict = self.workers.local_worker().learn_on_batch(samples)
            for policy_id, info in info_dict.items():
                self.learner_stats[policy_id] = get_learner_stats(info)
                replay_buffer = self.replay_buffers[policy_id]
                if isinstance(replay_buffer, PrioritizedReplayBuffer):
                    td_error = info["td_error"]
                    new_priorities = (
                        np.abs(td_error) + self.prioritized_replay_eps
                    )
                    replay_buffer.update_priorities(
                        samples.policy_batches[policy_id]["batch_indexes"],
                        new_priorities
                    )
            self.grad_timer.push_units_processed(samples.count)

        # Here!
        # self.num_steps_trained += samples.count
        self.num_steps_trained += np.mean(
            [b.count for b in samples.policy_batches.values()], dtype=np.int64
        )


def make_optimizer(workers, config):
    return SyncReplayOptimizerWithCooperativeExploration(
        workers,
        learning_starts=config["learning_starts"],
        buffer_size=config["buffer_size"],
        prioritized_replay=config["prioritized_replay"],
        prioritized_replay_alpha=config["prioritized_replay_alpha"],
        prioritized_replay_beta=config["prioritized_replay_beta"],
        schedule_max_timesteps=config["schedule_max_timesteps"],
        beta_annealing_fraction=config["beta_annealing_fraction"],
        final_prioritized_replay_beta=config["final_prioritized_replay_beta"],
        prioritized_replay_eps=config["prioritized_replay_eps"],
        train_batch_size=config["train_batch_size"],
        sample_batch_size=config["sample_batch_size"],
        **config["optimizer"]
    )


CETD3TFPolicy = DDPGTFPolicy

CETD3Trainer = TD3Trainer.with_updates(
    name="CETD3",
    default_config=cetd3_default_config,
    # default_policy=CETD3TFPolicy,
    # validate_config=validate_and_rewrite_config,
    make_policy_optimizer=make_optimizer
)
