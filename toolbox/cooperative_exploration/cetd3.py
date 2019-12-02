from ray.rllib.agents.ddpg.ddpg_policy import DDPGTFPolicy, SampleBatch
from ray.rllib.agents.ddpg.td3 import TD3Trainer, TD3_DEFAULT_CONFIG
from ray.rllib.optimizers.sync_replay_optimizer import SyncReplayOptimizer
from ray.tune.util import merge_dicts

DISABLE = "disable"
SHARE_SAMPLE = "share_sample"

cetd3_default_config = merge_dicts(
    TD3_DEFAULT_CONFIG, dict(
        mode=SHARE_SAMPLE
    )
    # dict(learn_with_peers=True, use_joint_dataset=False, mode=REPLAY_VALUES)
)


class SyncReplayOptimizerWithCooperativeExploration(SyncReplayOptimizer):

    def _replay(self):
        config = self.workers._local_config
        samples = super()._replay()
        if config["mode"] == SHARE_SAMPLE:
            share_sample = SampleBatch.concat_samples([
                other_batch
                for other, other_batch in samples.policy_batches.items()
            ])
            for pid in samples.policy_batches.keys():
                samples.policy_batches[pid] = share_sample
            samples.count = share_sample.count
        return samples


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
        **config["optimizer"])


CETD3TFPolicy = DDPGTFPolicy

CETD3Trainer = TD3Trainer.with_updates(
    name="CETD3",
    default_config=cetd3_default_config,
    # default_policy=CETD3TFPolicy,
    # validate_config=validate_and_rewrite_config,
    make_policy_optimizer=make_optimizer
)
