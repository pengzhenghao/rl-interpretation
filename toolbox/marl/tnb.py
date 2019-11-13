"""This file implement the modified version of TNB."""
from toolbox.marl.extra_loss_ppo_trainer import novelty_loss, \
    ppo_surrogate_loss, ExtraLossPPOTFPolicy, DEFAULT_CONFIG, merge_dicts, \
    ExtraLossPPOTrainer, validate_config

tnb_ppo_default_config = merge_dicts(DEFAULT_CONFIG, dict(
    joint_dataset_sample_batch_size=200
))


def tnb_loss(policy, model, dist_class, train_batch):
    """Add novelty loss with original ppo loss using TNB method"""
    original_loss = ppo_surrogate_loss(policy, model, dist_class, train_batch)
    nov_loss = novelty_loss(policy, model, dist_class, train_batch)

    # implement the logic of TNB
    total_loss = original_loss + nov_loss

    policy.total_loss = total_loss
    # return total_loss
    return [original_loss, nov_loss]


TNBPPOTFPolicy = ExtraLossPPOTFPolicy.with_updates(
    name="TNBPPOTFPolicy",
    get_default_config=lambda: tnb_ppo_default_config,
    loss_fn=tnb_loss,
)


def validate_config_tnb(config):
    assert "joint_dataset_sample_batch_size" in config
    validate_config(config)


TNBPPOTrainer = ExtraLossPPOTrainer.with_updates(
    name="TNBPPO",
    default_config=tnb_ppo_default_config,
    validate_config=validate_config_tnb,
    default_policy=TNBPPOTFPolicy
)

if __name__ == '__main__':
    from toolbox import initialize_ray
    from ray import tune
    from toolbox.utils import get_local_dir
    from toolbox.train.train_individual_marl import on_train_result
    from toolbox.marl.multiagent_env_wrapper import MultiAgentEnvWrapper

    num_agents = 5
    num_gpus = 0

    # This is only test code.
    initialize_ray(test_mode=True, local_mode=True, num_gpus=num_gpus)

    policy_names = ["ppo_agent{}".format(i) for i in range(num_agents)]

    env_config = {"env_name": "BipedalWalker-v2", "agent_ids": policy_names}
    env = MultiAgentEnvWrapper(env_config)
    config = {
        "env": MultiAgentEnvWrapper,
        "env_config": env_config,
        "num_gpus": num_gpus,
        "log_level": "DEBUG",
        "joint_dataset_sample_batch_size": 131,
        "multiagent": {
            "policies": {i: (None, env.observation_space, env.action_space, {})
                         for i in policy_names},
            "policy_mapping_fn": lambda x: x,
        },
        "callbacks": {
            "on_train_result": on_train_result
        },
    }

    tune.run(
        TNBPPOTrainer,
        local_dir=get_local_dir(),
        name="DELETEME_TEST_extra_loss_ppo_trainer",
        stop={"timesteps_total": 50000},
        config=config,
    )
