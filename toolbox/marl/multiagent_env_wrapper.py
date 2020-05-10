from itertools import count

import gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.normalize_actions import NormalizeActionWrapper

from toolbox.env import get_env_maker


def get_marl_env_config(env_name, num_agents, normalize_actions=False):
    config = {
        "env": MultiAgentEnvWrapper,
        "env_config": {
            "env_name": env_name,
            "num_agents": num_agents,
            "normalize_actions": normalize_actions
        },
        "multiagent": {}
    }
    return config


class MultiAgentEnvWrapper(MultiAgentEnv, gym.Env):
    """This is a brief wrapper to create a mock multi-agent environment"""

    def __init__(self, env_config):
        assert "num_agents" in env_config
        assert "env_name" in env_config
        num_agents = env_config['num_agents']
        agent_ids = ["agent{}".format(i) for i in range(num_agents)]
        self._render_policy = env_config.get('render_policy')
        self.num_agents = num_agents
        self.agent_ids = agent_ids
        self.env_name = env_config['env_name']
        self.env_maker = get_env_maker(
            env_config['env_name'], require_render=bool(self._render_policy)
        )

        if env_config.get("normalize_action", False):
            self.env_maker = lambda: NormalizeActionWrapper(self.env_maker())

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
        self.reward_range = tmp_env.reward_range
        self.metadata = tmp_env.metadata
        self.spec = tmp_env.spec

    def reset(self):
        self.dones = set()
        return {aid: env.reset() for aid, env in self.envs.items()}

    def step(self, action_dict):
        obs, rewards, dones, infos = {}, {}, {}, {}
        for aid, act in action_dict.items():
            # 0414 Workaround in dice-rebuttal
            # raise error is act is NaN
            if not np.all(np.isfinite(act)):
                raise ValueError(
                    "Agent {} input is not finite: {}".format(aid, act)
                )
            act = np.nan_to_num(act, copy=False)
            o, r, d, i = self.envs[aid].step(act)
            if d:
                if d in self.dones:
                    print(
                        "WARNING! current {} is already in"
                        "self.dones: {}. Given reward {}.".format(
                            aid, self.dones, r
                        )
                    )
                self.dones.add(aid)
            obs[aid], rewards[aid], dones[aid], infos[aid] = o, r, d, i
        dones["__all__"] = len(self.dones) == len(self.agent_ids)
        return obs, rewards, dones, infos

    def seed(self, s):
        for env_id, env in enumerate(self.envs.values()):
            env.seed(s + env_id * 10)

    def render(self, *args, **kwargs):
        assert self._render_policy
        assert self._render_policy in self.envs, (
            self._render_policy, self.envs.keys()
        )
        return self.envs[self._render_policy].render(*args, **kwargs)

    def __repr__(self):
        return "MultiAgentEnvWrapper({})".format(self.env_name)


if __name__ == '__main__':
    import time

    env_config = get_marl_env_config("CartPole-v0", 10)["env_config"]
    mae = MultiAgentEnvWrapper(env_config)
    alive = set(mae.agent_ids)
    mae.reset()
    for i in count():
        time.sleep(0.05)
        acts = {a: np.random.randint(2) for a in alive}
        ret = mae.step(acts)
        print(
            "At timestep {}, the applied action is {} and the return is {}"
            "".format(i, acts, ret)
        )
        for dead_id, dead in ret[2].items():
            if dead_id == "__all__" or (not dead):
                continue
            alive.remove(dead_id)
        if ret[2]['__all__']:
            print("Finish! ", ret)
            break
