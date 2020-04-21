import logging

import gym
import numpy as np
from gym import ActionWrapper
from ray.tune.registry import register_env

logger = logging.getLogger(__file__)

msg = """
****************************************
****************************************
****************************************
****************************************
****************************************
WARNING! We found {}:
{}
****************************************
****************************************
****************************************
****************************************
****************************************
"""


class WarnActionWrapper(ActionWrapper):
    def action(self, action):
        action = np.asarray(action)
        if not np.all(np.isfinite(action)):
            tmp = msg.format("action is not finite", action)
            print(tmp)
            logger.error(tmp)
        elif not self.env.action_space.contains(action):
            tmp = msg.format("action out of bound", action)
            print(tmp)
            logger.error(tmp)
        return action


def env_creator(_=None):
    env = gym.make("BipedalWalker-v2")
    env = WarnActionWrapper(env)
    return env


def register():
    register_env("WrappedBipedalWalker-v2", env_creator)
    print("WrappedBipedalWalker-v2 registered!")


register()
