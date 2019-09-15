import gym
from gym.envs.box2d.bipedal_walker import BipedalWalker

from toolbox.env.mujoco_wrapper import MujocoWrapper


def build_bipedal_walker(useless=None):
    env = gym.make("BipedalWalker-v2")
    env.seed(0)
    return env


def build_halfcheetah(useless=None):
    env = gym.make("HalfCheetah-v2")
    env.seed(0)
    env = MujocoWrapper(env)
    return env


def get_env_maker(name):
    return ENV_MAKER_LOOKUP[name]


ENV_MAKER_LOOKUP = {
    "BipedalWalker-v2": build_bipedal_walker,
    "HalfCheetah-v2": build_halfcheetah
}
