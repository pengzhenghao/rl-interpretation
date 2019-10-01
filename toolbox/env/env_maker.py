import gym

from toolbox.env.env_wrapper import BipedalWalkerWrapper

try:
    from toolbox.env.mujoco_wrapper import MujocoWrapper, \
        HalfCheetahV2NoBackground, HalfCheetahV3NoBackground, \
        HopperV3NoBackground, Walker2dV3NoBackground
except gym.error.DependencyNotInstalled:
    print("Filed to import mujoco environment wrapper. "
          "This may because you didn't install mujoco_py.")

DEFAULT_SEED = 0


def build_bipedal_walker(useless=None):
    env = gym.make("BipedalWalker-v2")
    env.seed(DEFAULT_SEED)
    return env


def build_opencv_bipedal_walker(useless=None):
    env = BipedalWalkerWrapper()
    env.seed(DEFAULT_SEED)
    return env


def build_halfcheetahv2(require_shadow=False):
    # env = gym.make("HalfCheetah-v2")
    env = HalfCheetahV2NoBackground(require_shadow)
    env.seed(DEFAULT_SEED)
    env = MujocoWrapper(env)
    return env


def build_halfcheetahv2_shadow(require_shadow=True):
    # env = gym.make("HalfCheetah-v2")
    env = HalfCheetahV2NoBackground(require_shadow)
    env.seed(DEFAULT_SEED)
    env = MujocoWrapper(env)
    return env


def build_halfcheetahv3(require_shadow=False):
    # env = gym.make("HalfCheetah-v2")
    env = HalfCheetahV3NoBackground(require_shadow)
    env.seed(DEFAULT_SEED)
    env = MujocoWrapper(env)
    return env


def build_hopperv3(require_shadow=False):
    # env = gym.make("HalfCheetah-v2")
    env = HopperV3NoBackground(require_shadow)
    env.seed(DEFAULT_SEED)
    env = MujocoWrapper(env)
    return env


def build_walkerv3(require_shadow=False):
    # env = gym.make("HalfCheetah-v2")
    env = Walker2dV3NoBackground(require_shadow)
    env.seed(DEFAULT_SEED)
    env = MujocoWrapper(env)
    return env

from gym.envs.mujoco.walker2d_v3 import Walker2dEnv

def get_env_maker(name, require_render=False):
    if require_render and name == "BipedalWalker-v2":
        return build_opencv_bipedal_walker
    if require_render and name == "Walker2d-v3":
        return Walker2dEnv
    return ENV_MAKER_LOOKUP[name]


ENV_MAKER_LOOKUP = {
    "BipedalWalker-v2": build_bipedal_walker,
    "HalfCheetah-v2-shadow": lambda: build_halfcheetahv2(True),
    "HalfCheetah-v2": build_halfcheetahv2,
    "HalfCheetah-v3": build_halfcheetahv3,
    "HalfCheetah-v3-shadow": lambda: build_halfcheetahv3(True),
    "Hopper-v3": build_hopperv3,
    "Hopper-v3-shadow": lambda: build_hopperv3(True),
    "Walker2d-v3": build_walkerv3,
    "Walker2d-v3-shadow": lambda: build_walkerv3(True)
}
