# from gym.envs.mujoco.walker2d import Walker2dEnv
import gym
import numpy as np
from gym.core import Wrapper
import os
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv

class HalfCheetahEnvNoBackground(HalfCheetahEnv):
    def __init__(self, require_shadow=False):
        xml_path = 'half_cheetah(no_shadow).xml'
        if require_shadow:
            xml_path = "half_cheetah.xml"
        xml_path = os.path.join(os.path.dirname(__file__),
                                "modified_mujoco_assets", xml_path)
        super(HalfCheetahEnvNoBackground, self).__init__(xml_file=xml_path)


class MujocoWrapper(Wrapper):
    def get_state_wrap(self):
        return self.sim.get_state()

    def set_state_wrap(self, state):
        self.sim.set_state(state)


def _test_MujocoWrapper(env_name):
    env = gym.make(env_name)
    env = MujocoWrapper(env)

    env.seed(0)
    env.reset()

    states = []
    acts = []
    for _ in range(10):
        act = env.action_space.sample()
        acts.append(act)
        env.step(act)
        state = env.get_state_wrap()
        states.append(state)

    # Test if replicable
    new_env = gym.make(env_name)
    new_env = MujocoWrapper(new_env)
    new_env.seed(0)
    new_env.reset()

    for act, state in zip(acts, states):
        new_env.step(act)
        current_state = new_env.get_state_wrap()
        assert state == current_state

    # restore old env
    env.seed(0)
    env.reset()
    env.set_state_wrap(states[5])
    env.step(acts[6])
    current_state = env.get_state_wrap()
    assert current_state.time == states[6].time
    np.testing.assert_array_almost_equal(current_state.qpos, states[6].qpos)
    np.testing.assert_array_almost_equal(current_state.qvel, states[6].qvel)
    np.testing.assert_array_equal(current_state.act, states[6].act)

    print("Successfully Pass Test for env: {}!!!".format(env_name))
    return current_state, states[6]


def test_MujocoWrapper():
    env_name_list = ["Ant-v2", "HalfCheetah-v2", "Humanoid-v2", "Walker2d-v2"]
    for env_name in env_name_list:
        _test_MujocoWrapper(env_name)


if __name__ == '__main__':
    test_MujocoWrapper()
