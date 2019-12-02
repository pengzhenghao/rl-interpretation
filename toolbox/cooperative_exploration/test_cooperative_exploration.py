from toolbox.cooperative_exploration.cooperative_exploration_ppo import \
    CEPPOTrainer, ceppo_default_config


def debug_ceppo(local_mode):
    from toolbox.marl.test_extra_loss import _base

    _base(
        CEPPOTrainer,
        local_mode,
        extra_config={"disable": True},
        env_name="CartPole-v0"
    )


def validate_ceppo(disable, test_mode=False):
    from ray import tune
    from toolbox import initialize_ray
    from toolbox.marl import MultiAgentEnvWrapper

    initialize_ray(test_mode=test_mode, local_mode=False)

    env_name = "CartPole-v0"
    num_agents = 3
    policy_names = ["ppo_agent{}".format(i) for i in range(num_agents)]
    env_config = {"env_name": env_name, "agent_ids": policy_names}
    env = MultiAgentEnvWrapper(env_config)
    config = {
        "seed": 0,
        "env": MultiAgentEnvWrapper,
        "env_config": env_config,
        "multiagent": {
            "policies": {
                i: (None, env.observation_space, env.action_space, {})
                for i in policy_names
            },
            "policy_mapping_fn": lambda x: x,
        },
        "disable": disable,
    }

    if disable:
        config['train_batch_size'] = \
            ceppo_default_config['train_batch_size'] * num_agents
        config['num_workers'] = \
            ceppo_default_config['num_workers'] * num_agents

    tune.run(
        CEPPOTrainer,
        name="DELETEME_TEST_CEPPO",
        # stop={"timesteps_total": 50000},
        stop={"info/num_steps_trained": 50000},
        config=config
    )


if __name__ == '__main__':
    # debug_ceppo(local_mode=False)
    validate_ceppo(disable=False, test_mode=False)
