import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from toolbox.env import get_env_maker


class MultiAgentEnvWrapper(MultiAgentEnv):
    """This is a brief wrapper to create a mock multi-agent environment"""

    def __init__(self, env_config):
        assert "num_agents" in env_config
        assert "env_name" in env_config
        num_agents = env_config['num_agents']
        agent_ids = ["agent{}".format(i) for i in range(num_agents)]
        self.num_agents = num_agents
        self.agent_ids = agent_ids
        self.env_name = env_config['env_name']
        self.env_maker = get_env_maker(env_config['env_name'])
        self.envs = {}
        if not isinstance(agent_ids, list):
            agent_ids = [agent_ids]
        for aid in agent_ids:
            if aid not in self.envs:
                self.envs[aid] = self.env_maker()
        self.dones = set()
        tmp_env = next(iter(self.envs.values()))
        self.observation_space = tmp_env.observation_space
        self.action_space = tmp_env.action_space

    def reset(self):
        self.dones = set()
        return {aid: env.reset() for aid, env in self.envs.items()}

    def step(self, action_dict):
        obs, rewards, dones, infos = {}, {}, {}, {}
        for aid, act in action_dict.items():
            act = np.nan_to_num(act, copy=False)
            o, r, d, i = self.envs[aid].step(act)
            if d:
                self.dones.add(aid)
            obs[aid], rewards[aid], dones[aid], infos[aid] = o, r, d, i
        dones["__all__"] = len(self.dones) == len(self.agent_ids)
        return obs, rewards, dones, infos

    def seed(self, s):
        for env in self.envs.values():
            env.seed(s)

    def __repr__(self):
        return "MultiAgentEnvWrapper({})".format(self.env_name)


if __name__ == '__main__':

    env_config = {"env_name": "BipedalWalker-v2", "agent_ids": list(range(10))}

    mae = MultiAgentEnvWrapper(env_config)
    mae.reset()
    while True:
        ret = mae.step({i: np.zeros((17,)) for i in range(10)})
        if ret[2]['__all__']:
            print("Finish! ", ret)
            break
