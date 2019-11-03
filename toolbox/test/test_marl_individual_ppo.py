from ray import tune

from toolbox import initialize_ray
from toolbox.env import get_env_maker
from toolbox.marl.multiagent_env_wrapper import MultiAgentEnvWrapper


def test_marl_individual_ppo():
    num_gpus = 4
    exp_name = "test_marl_individual_ppo"
    env_name = "BipedalWalker-v2"
    num_iters = 50
    num_agents = 8

    initialize_ray(test_mode=True, num_gpus=num_gpus)

    tmp_env = get_env_maker(env_name)()

    default_policy = (
        None,
        tmp_env.observation_space,
        tmp_env.action_space,
        {}
    )

    policy_names = ["Agent{}".format(i) for i in range(num_agents)]

    def policy_mapping_fn(aid):
        print("input aid: ", aid)
        return aid

    tune.run(
        "PPO",
        name=exp_name,
        checkpoint_at_end=True,
        checkpoint_freq=10,
        stop={"training_iteration": num_iters},
        config={
            "env": MultiAgentEnvWrapper,
            "env_config": {
                "env_name": env_name,
                "agent_ids": policy_names
            },
            "log_level": "DEBUG",
            "num_gpus": num_gpus,
            "multiagent": {
                "policies": {
                    i: default_policy for i in policy_names
                },
                "policy_mapping_fn": policy_mapping_fn,
            },
        },
    )


if __name__ == '__main__':
    test_marl_individual_ppo()
