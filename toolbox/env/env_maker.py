import gym

from toolbox.env.env_wrapper import BipedalWalkerWrapper
from toolbox.env.mujoco_wrapper import MujocoWrapper, \
    HalfCheetahV2NoBackground, HalfCheetahV3NoBackground

DEFAULT_SEED = 0


def build_bipedal_walker(useless=None):
    env = gym.make("BipedalWalker-v2")
    env.seed(DEFAULT_SEED)
    return env


def build_opencv_bipedal_walker(useless=None):
    env = BipedalWalkerWrapper()
    env.seed(DEFAULT_SEED)
    return env


def build_halfcheetahv2(useless=None):
    # env = gym.make("HalfCheetah-v2")
    env = HalfCheetahV2NoBackground()
    env.seed(DEFAULT_SEED)
    env = MujocoWrapper(env)
    return env


def build_halfcheetahv3(useless=None):
    # env = gym.make("HalfCheetah-v2")
    env = HalfCheetahV3NoBackground()
    env.seed(DEFAULT_SEED)
    env = MujocoWrapper(env)
    return env


def get_env_maker(name, require_render=False):
    if require_render and name == "BipedalWalker-v2":
        return build_opencv_bipedal_walker
    return ENV_MAKER_LOOKUP[name]


ENV_MAKER_LOOKUP = {
    "BipedalWalker-v2": build_bipedal_walker,
    "HalfCheetah-v2": build_halfcheetahv2,
    "HalfCheetah-v3": build_halfcheetahv3
}
