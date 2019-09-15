import sys
sys.path.append("../")
from toolbox.env.mujoco_wrapper import MujocoWrapper
import numpy as np


def test_MujocoWrapper():
    from gym.envs.mujoco.walker2d import Walker2dEnv
    env = Walker2dEnv()
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
    new_env = Walker2dEnv()
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
    print("Successfully Pass Test!!!")
    # assert 1==0
    assert current_state.time == states[6].time
    np.testing.assert_array_almost_equal(current_state.qpos, states[6].qpos)
    np.testing.assert_array_almost_equal(current_state.qvel, states[6].qvel)
    np.testing.assert_array_equal(current_state.act, states[6].act)
    return current_state, states[6]
