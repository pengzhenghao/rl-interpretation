from toolbox.marl.multiagent_env_wrapper import MultiAgentEnvWrapper
from toolbox.marl.utils import on_train_result


def get_marl_env_config(env_name, num_agents):
    config = {
        "env": MultiAgentEnvWrapper,
        "env_config": {"env_name": env_name, "num_agents": num_agents},
        "multiagent": {}
    }
    return config
