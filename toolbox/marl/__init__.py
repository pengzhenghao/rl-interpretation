from ray.tune.registry import _global_registry, ENV_CREATOR

from toolbox.marl.multiagent_env_wrapper import MultiAgentEnvWrapper
from toolbox.marl.utils import on_train_result


def get_marl_env_config(env_name, num_agents):
    config = {
        "env": MultiAgentEnvWrapper,
        "env_config": {"env_name": env_name, "num_agents": num_agents}
    }

    assert _global_registry.contains(ENV_CREATOR, config["env"])
    env_creator = _global_registry.get(ENV_CREATOR, config["env"])
    tmp_env = env_creator(config["env_config"])
    config["multiagent"]["policies"] = {
        "agent{}".format(i): (
            None,
            tmp_env.observation_space,
            tmp_env.action_space,
            {}
        )
        for i in range(num_agents)
    }
    config["multiagent"]["policy_mapping_fn"] = lambda x: x

    return config
