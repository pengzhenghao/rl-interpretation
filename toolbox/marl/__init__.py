from toolbox.marl.multiagent_env_wrapper import MultiAgentEnvWrapper
from toolbox.marl.utils import on_train_result


def get_marl_env_config(env_name, num_agents):
    config = {
        "env": MultiAgentEnvWrapper,
        "env_config": {"env_name": env_name, "num_agents": num_agents},
        "multiagent": {}
    }
    tmp_env = MultiAgentEnvWrapper(config["env_config"])
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
