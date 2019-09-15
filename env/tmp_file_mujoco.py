from gym.envs.mujoco.walker2d import Walker2dEnv


class Walker2dEnvWrapper(Walker2dEnv):

    def get_state(self):
        return self.sim.get_state()

    def set_state_wrap(self, state):
        # self.set_state()
        pass

if __name__ == '__main__':
    ww = Walker2dEnvWrapper()
    state = ww.get_state()
    print(state)