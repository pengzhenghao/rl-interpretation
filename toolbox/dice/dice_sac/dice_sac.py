from ray.rllib.agents.sac.sac import SACTrainer, \
    validate_config as validate_config_sac
from ray.rllib.optimizers.sync_replay_optimizer import SyncReplayOptimizer
from ray.tune.registry import _global_registry, ENV_CREATOR

from toolbox.dice.dice import setup_policies_pool
from toolbox.dice.dice_postprocess import MY_LOGIT
from toolbox.dice.dice_sac.dice_sac_config import dice_sac_default_config
from toolbox.dice.dice_sac.dice_sac_policy import DiCESACPolicy


# from toolbox.dice.dice_sac.dice_sac_optimizer import
# SyncReplayOptimizerModified


# def after_optimizer_step(trainer, fetches):
#     # Original SAC operation
#     update_target_if_needed(trainer, fetches)

# only update the policies pool if used DELAY_UPDATE, otherwise
# the policies_pool in each policy is simply not used, so we don't
# need to update it.
# if trainer.config[constants.DELAY_UPDATE]:
# if trainer.workers.remote_workers():
#     weights = ray.put(trainer.workers.local_worker().get_weights())
#     for e in trainer.workers.remote_workers():
#         e.set_weights.remote(weights)
#
#     def _delay_update_for_worker(worker, worker_index):
#         worker.foreach_policy(lambda p, _: p.update_target_network())
#
#     trainer.workers.foreach_worker_with_index(_delay_update_for_worker)


def validate_config(config):
    validate_config_sac(config)

    # Hard-coded this setting
    assert not config["normalize_actions"]
    assert config["env_config"]["normalize_actions"]

    # create multi-agent environment
    assert _global_registry.contains(ENV_CREATOR, config["env"])
    env_creator = _global_registry.get(ENV_CREATOR, config["env"])
    tmp_env = env_creator(config["env_config"])
    config["multiagent"]["policies"] = {
        i: (None, tmp_env.observation_space, tmp_env.action_space, {})
        for i in tmp_env.agent_ids
    }
    config["multiagent"]["policy_mapping_fn"] = lambda x: x
    config['model']['custom_model'] = None
    config['model']['custom_options'] = None


def before_learn_on_batch(samples, policy_map, train_batch_size):
    for agent_id, my_batch in samples.policy_batches.items():
        assert agent_id in policy_map, (agent_id, policy_map.keys())
        my_batch[MY_LOGIT] = policy_map[agent_id] \
            ._compute_my_deterministic_action(my_batch["obs"])
        samples.policy_batches[agent_id]["diversity_rewards"] = \
            policy_map[agent_id].compute_diversity(
                my_batch,
                {pid: p for pid, p in policy_map.items() if pid != agent_id}
            )
    return samples


def make_policy_optimizer(workers, config):
    """Create the single process DQN policy optimizer.

    Returns:
        SyncReplayOptimizer: Used for generic off-policy Trainers.
    """
    return SyncReplayOptimizer(
        workers,
        learning_starts=config["learning_starts"],
        buffer_size=config["buffer_size"],
        prioritized_replay=config["prioritized_replay"],
        prioritized_replay_alpha=config["prioritized_replay_alpha"],
        prioritized_replay_beta=config["prioritized_replay_beta"],
        prioritized_replay_beta_annealing_timesteps=config[
            "prioritized_replay_beta_annealing_timesteps"],
        final_prioritized_replay_beta=config["final_prioritized_replay_beta"],
        prioritized_replay_eps=config["prioritized_replay_eps"],
        train_batch_size=config["train_batch_size"],
        before_learn_on_batch=before_learn_on_batch,  # <<==
        **config["optimizer"]
    )


DiCESACTrainer = SACTrainer.with_updates(
    name="DiCESACTrainer",
    default_config=dice_sac_default_config,
    default_policy=DiCESACPolicy,
    get_policy_class=lambda _: DiCESACPolicy,
    after_init=setup_policies_pool,
    # after_optimizer_step=update_target_if_needed,
    validate_config=validate_config,
    make_policy_optimizer=make_policy_optimizer
)
