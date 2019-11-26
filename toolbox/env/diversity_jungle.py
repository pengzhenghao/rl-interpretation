import copy
import matplotlib
matplotlib.use("MacOSX")

import matplotlib.pyplot as plt
import numpy as np
import time


class FourWayGridWorld:
    def __init__(self, N=17, left=10, right=10, up=10, down=10):
        self.N = 17
        self.left = left
        self.right = right
        self.up = up
        self.down = down
        self.map = np.ones((N, N)) * (-0.1)
        self.map[int((N - 1) / 2), 0] = self.left
        self.map[0, int((N - 1) / 2)] = self.up
        self.map[N - 1, int((N - 1) / 2)] = self.down
        self.map[int((N - 1) / 2), N - 1] = self.right
        self.loc = np.asarray([np.random.randint(N), np.random.randint(N)])
        self.step_num = 0
        self.init_render = False

    def step(self, action):
        action = np.clip(action, -1, 1)
        new_loc = np.clip(self.loc + action, 0, self.N - 1)
        # if self.map[self.loc[0],self.loc[1]]!=0:
        self.loc = new_loc
        reward = self.map[int(self.loc[0]), int(self.loc[1])]
        self.step_num += 1
        return self.loc, reward, self.ifdone()

    def ifdone(self):
        if self.step_num >= 2 * self.N:
            return True
        else:
            return False

    def render(self):
        map_self = copy.deepcopy(self.map)
        map_self[int(self.loc[0]), int(self.loc[1])] = -5
        if not self.init_render:
            self.canvas = plt.imshow(map_self, animated=True)
        self.canvas.set_data(map_self)
        plt.draw()

    def reset(self):
        self.map = np.ones((self.N, self.N)) * (-0.1)
        self.map[int((self.N - 1) / 2), 0] = self.left
        self.map[0, int((self.N - 1) / 2)] = self.up
        self.map[self.N - 1, int((self.N - 1) / 2)] = self.down
        self.map[int((self.N - 1) / 2), self.N - 1] = self.right
        self.loc = np.asarray(
            [np.random.randint(self.N), np.random.randint(self.N)])
        self.step_num = 0
        return self.loc


if __name__ == '__main__':
    env = FourWayGridWorld()
    obs = env.reset()
    for t in range(100):
        o, r, d = env.step(np.random.normal(size=2))
        print("Current t {}, observation {}, reward {}, done {}.".format(
            t, o, r, d)
        )
        env.render()
        time.sleep(0.2)
        if d:
            break
