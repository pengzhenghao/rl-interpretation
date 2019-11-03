from ray.rllib.env.multi_agent_env import MultiAgentEnv


class MultiAgentEnvWrapper(MultiAgentEnv):
    """This is a brief wrapper to create a mock multi-agent environment"""
    def __init__(self, agent_ids, env_maker):
        if not isinstance(agent_ids, list):
            agent_ids = [agent_ids]
        assert callable(env_maker)
        self.env_maker = env_maker
        self.envs = {aid: self.env_maker() for aid in agent_ids}

    def add(self, agent_ids):
        if not isinstance(agent_ids, list):
            agent_ids = [agent_ids]
        for aid in agent_ids:
            if aid not in self.envs:
                self.envs[aid] = self.env_maker()

    def reset(self):
        for env in self.envs.values():
            env.reset()

    def step(self, action_dict):
        obs, rewards, dones, infos = {}, {}, {}, {}
        for aid, act in action_dict.items():
            o, r, d, i = self.envs[aid].step(act)
            obs[aid], rewards[aid], dones[aid], infos[aid] = o, r, d, i
        dones["__all__"] = all(dones.values())
        return obs, rewards, dones, infos


if __name__ == '__main__':
    from toolbox.env import get_env_maker
    import numpy as np

    mae = MultiAgentEnvWrapper(
        list(range(10)), get_env_maker("BipedalWalker-v2"))
    mae.reset()
    while True:
        ret = mae.step({i: np.zeros((17,)) for i in range(10)})
        if ret[2]['__all__']:
            print("Finish! ", ret)
            break
