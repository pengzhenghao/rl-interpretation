from ray.rllib.env.multi_agent_env import MultiAgentEnv

from toolbox.env import get_env_maker


class MultiAgentEnvWrapper(MultiAgentEnv):
    """This is a brief wrapper to create a mock multi-agent environment"""

    def __init__(self, env_config):
        assert "agent_ids" in env_config
        assert "env_name" in env_config
        agent_ids = env_config['agent_ids']
        self.agent_ids = agent_ids
        self.env_maker = get_env_maker(env_config['env_name'])
        self.envs = {}
        self.add(agent_ids)

        tmp_env = next(iter(self.envs.values()))
        self.observation_space = tmp_env.observation_space
        self.action_space = tmp_env.action_space

    def add(self, agent_ids):
        assert agent_ids
        if not isinstance(agent_ids, list):
            agent_ids = [agent_ids]
        for aid in agent_ids:
            assert aid in self.agent_ids
            if aid not in self.envs:
                self.envs[aid] = self.env_maker()

    def reset(self):
        obs = {}
        for aid, env in self.envs.items():
            o = env.reset()
            obs[aid] = o
        return obs

    def step(self, action_dict):
        obs, rewards, dones, infos = {}, {}, {}, {}
        for aid, act in action_dict.items():
            assert aid in self.agent_ids
            assert aid in self.envs
            o, r, d, i = self.envs[aid].step(act)
            obs[aid], rewards[aid], dones[aid], infos[aid] = o, r, d, i
        dones["__all__"] = all(dones.values())
        return obs, rewards, dones, infos


# def make_multiagent_env_wrapper(env_maker):
#     class Env(MultiAgentEnvWrapper):
#         def __init__(self, agent_ids):
#             super(Env, self).__init__(agent_ids, env_maker)
#
#     return Env

if __name__ == '__main__':
    import numpy as np

    env_config = {"env_name": "BipedalWalker-v2", "agent_ids": list(range(10))}

    mae = MultiAgentEnvWrapper(env_config)
    mae.reset()
    while True:
        ret = mae.step({i: np.zeros((17, )) for i in range(10)})
        if ret[2]['__all__']:
            print("Finish! ", ret)
            break
