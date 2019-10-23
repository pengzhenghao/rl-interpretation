import gym

from toolbox.env.bipedal_walker_wrapper import BipedalWalkerWrapper

try:
    from toolbox.env.mujoco_wrapper import MujocoWrapper, \
        HalfCheetahV2NoBackground, HalfCheetahV3NoBackground, \
        HopperV3NoBackground, Walker2dV3NoBackground
except gym.error.DependencyNotInstalled:
    print(
        "Filed to import mujoco environment wrapper. "
        "This may because you didn't install mujoco_py."
    )

DEFAULT_SEED = 0


def make_build_gym_env(env_name):
    def build_gym_env(seed=None):
        if seed is None:
            seed = DEFAULT_SEED
        env = gym.make(env_name)
        env.seed(seed)
        return env

    return build_gym_env


def build_opencv_bipedal_walker(seed=None):
    if seed is None:
        seed = DEFAULT_SEED
    env = BipedalWalkerWrapper()
    env.seed(seed)
    return env


def _build_mujoco_env(cls, seed, require_shadow):
    if seed is None:
        seed = DEFAULT_SEED
    env = cls(require_shadow)
    env.seed(seed)
    env = MujocoWrapper(env)
    return env


def build_halfcheetahv2(seed=None, require_shadow=False):
    return _build_mujoco_env(HalfCheetahV2NoBackground, seed, require_shadow)


def build_halfcheetahv2_shadow(seed=None, require_shadow=True):
    return _build_mujoco_env(HalfCheetahV2NoBackground, seed, require_shadow)


def build_halfcheetahv3(seed=None, require_shadow=False):
    return _build_mujoco_env(HalfCheetahV3NoBackground, seed, require_shadow)


def build_hopperv3(seed=None, require_shadow=False):
    return _build_mujoco_env(HopperV3NoBackground, seed, require_shadow)


def build_walkerv3(seed=None, require_shadow=False):
    return _build_mujoco_env(Walker2dV3NoBackground, seed, require_shadow)


def get_env_maker(name, require_render=False):
    if require_render and name == "BipedalWalker-v2":
        return build_opencv_bipedal_walker
    if require_render:
        return make_build_gym_env(name)
    else:
        return ENV_MAKER_LOOKUP[name]


ENV_MAKER_LOOKUP = {
    "BipedalWalker-v2": make_build_gym_env("BipedalWalker-v2"),
    "HalfCheetah-v2-shadow": lambda: build_halfcheetahv2(True),
    "HalfCheetah-v2": build_halfcheetahv2,
    "HalfCheetah-v3": build_halfcheetahv3,
    "HalfCheetah-v3-shadow": lambda: build_halfcheetahv3(True),
    "Hopper-v3": build_hopperv3,
    "Hopper-v3-shadow": lambda: build_hopperv3(True),
    "Walker2d-v3": build_walkerv3,
    "Walker2d-v3-shadow": lambda: build_walkerv3(True)
}
