import gym
import numpy as np

from toolbox.env.bipedal_walker_wrapper import BipedalWalkerWrapper

try:
    from toolbox.env.mujoco_wrapper import MujocoWrapper, \
        HalfCheetahV2NoBackground, HalfCheetahV3NoBackground, \
        HopperV3NoBackground, Walker2dV3NoBackground
except Exception:
    print(
        "Filed to import mujoco environment wrapper. "
        "This may because you didn't install mujoco_py."
    )

try:
    import pybullet_envs
except Exception:
    print("Failed to import pybullet_envs!")

try:
    import gym_minigrid
except ImportError:
    print("Failed to import minigrid environments!")


class MiniGridWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(MiniGridWrapper, self).__init__(env)
        space = self.env.observation_space.spaces["image"]
        length = np.prod(space.shape)
        shape = [length, ]
        self.observation_space = gym.spaces.Box(
            low=space.low.reshape(-1)[0],
            high=space.high.reshape(-1)[0],
            shape=shape
        )

    def observation(self, obs):
        return obs["image"].ravel()


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
    if require_render and name in ENV_MAKER_LOOKUP:
        return make_build_gym_env(name)
    if callable(name):
        return lambda: name()
    # if name in ENV_MAKER_LOOKUP:
    #     return ENV_MAKER_LOOKUP[name]
    if isinstance(name, str) and name.startswith("MiniGrid"):
        print("Return the mini grid environment {} with MiniGridWrapper("
              "FlatObsWrapper)!".format(
            name))
        return lambda: MiniGridWrapper(gym.make(name))
    else:
        assert name in [s.id for s in gym.envs.registry.all()], \
            "name of env not in {}".format(
                [s.id for s in gym.envs.registry.all()])
        return lambda: gym.make(name)


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
