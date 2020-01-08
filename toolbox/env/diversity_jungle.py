import gym
import numpy as np
from gym.spaces import Box


class Wall:
    def __init__(self, p1, p2):
        self.start = tuple(p1)
        self.end = tuple(p2)

    def intersect(self, p3, p4):
        """Return True if line (self.start, self.end) intersect with line
        (p3, p4)"""
        p1 = self.start
        p2 = self.end
        vx = p2[0] - p1[0]
        vy = p2[1] - p1[1]
        v1x = p3[0] - p1[0]
        v1y = p3[1] - p1[1]
        v2x = p4[0] - p1[0]
        v2y = p4[1] - p1[1]
        if (vx * v1y - vy * v1x) * (vx * v2y - vy * v2x) > 1e-6:
            return False
        vx = p4[0] - p3[0]
        vy = p4[1] - p3[1]
        v1x = p1[0] - p3[0]
        v1y = p1[1] - p3[1]
        v2x = p2[0] - p3[0]
        v2y = p2[1] - p3[1]
        if (vx * v1y - vy * v1x) * (vx * v2y - vy * v2x) > 1e-6:
            return False
        return True


class FourWayGridWorld(gym.Env):
    def __init__(self, env_config=None):
        self.N = 16
        self.observation_space = Box(0, self.N, shape=(2,))
        self.action_space = Box(-1, 1, shape=(2,))

        self.config = env_config
        self.early_done = env_config.get(
            'early_done'
        ) if env_config is not None else False
        self.int_initialize = not env_config.get(
            'not_int_initialize'
        ) if env_config is not None else True
        self.use_walls = env_config.get(
            'use_walls'
        ) if env_config is not None else False

        self.map = np.ones((self.N + 1, self.N + 1), dtype=np.float32) * (-0.1)
        self._fill_map()

        self.walls = []
        self._fill_walls()
        self.reset()

    def _fill_walls(self):
        """Let suppose you have three walls, two vertical, one horizontal."""
        if self.use_walls:
            print('Building Walls!!!')
            self.walls.append(Wall([4, 6], [12, 6]))
            self.walls.append(Wall([4, 10], [12, 10]))
            self.walls.append(Wall([12, 6], [12, 10]))

    def _fill_map(self):
        left = 10
        right = 10
        up = 10
        down = 10
        self.map[int(self.N / 2), 0] = left
        self.map[0, int(self.N / 2)] = up
        self.map[self.N, int(self.N / 2)] = down
        self.map[int(self.N / 2), self.N] = right
        self.traj = []

    @property
    def done(self):
        if self.early_done:
            return (not (0 <= self.loc[0] <= self.N - 1)) or \
                   (not (0 <= self.loc[1] <= self.N - 1)) or \
                   (self.step_num >= 2 * self.N)
        return self.step_num >= 2 * self.N

    def step(self, action):
        action = np.clip(action, -1, 1).astype(np.float32)
        new_loc = np.clip(self.loc + action, 0, self.N)
        if any(w.intersect(self.loc, new_loc) for w in self.walls):
            pass
        else:
            self.loc = new_loc
        reward = self.map[int(round(self.loc[0])), int(round(self.loc[1]))]
        self.step_num += 1
        self.traj.append(self.loc)
        return self.loc, reward, self.done, {}

    def render(self, mode=None):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        img = ax.imshow(
            self.map, aspect=1,
            extent=[-0.5, self.N + 0.5, -0.5, self.N + 0.5], cmap=plt.cm.hot_r
        )
        fig.colorbar(img)
        ax.set_aspect(1)
        for w in self.walls:
            x = [w.start[0], w.end[0]]
            y = [w.start[1], w.end[1]]
            ax.plot(x, y, c='orange')
        if len(self.traj) > 0:
            traj = np.asarray(self.traj)
            ax.plot(traj[:, 0], traj[:, 1], c='blue', alpha=0.75)
        ax.set_xlabel('X-coordinate')
        ax.set_ylabel('Y-coordinate')
        ax.set_xlim(0, self.N)
        ax.set_ylim(0, self.N)
        if self.config["_show"]:
            plt.show()
        return fig, ax

    def reset(self):
        if self.int_initialize:
            self.loc = np.random.randint(
                0, self.N + 1, size=(2,)
            ).astype(np.float32)
        else:
            self.loc = np.random.uniform(
                0, self.N, size=(2,)
            ).astype(np.float32)
        self.step_num = 0
        self.traj = []
        return self.loc

    def seed(self, s=None):
        if s is not None:
            np.random.seed(s)


def draw(compute_action, env_config):
    """compute_action is a function that take current obs (array with shape
    (2,)) as input and return the action: array with shape (2,)."""
    import matplotlib.pyplot as plt
    env_config['_show'] = False
    env = FourWayGridWorld(env_config)
    fig, ax = env.render()
    for i in range(17):
        for j in range(17):
            action = compute_action([i, j])
            ax.arrow(i, j, action[0], action[1], head_width=0.2, shape='left')
    plt.show()


def _test_draw():
    compute_action = lambda _: (1, 0.5)
    draw(compute_action, {})


def _debug_plot(map):
    """map is an 2D array."""
    import matplotlib.pyplot as plt
    plt.imshow(map, cmap=plt.cm.hot_r)
    plt.show()


def _test_line_intersect():
    wall = Wall([0, 0], [10, 10])
    assert wall.intersect([0, 10], [10, 0])
    assert wall.intersect([0, 10], [0, 0])
    assert wall.intersect([-10, 10], [10, -10])
    assert not wall.intersect([-10, 10], [10, -11])
    for i in range(10):
        Wall(
            np.random.randint(-10000, 10000, size=(2,)),
            np.random.randint(-10000, 10000, size=(2,))
        ).intersect(
            np.random.randint(-10000, 10000, size=(2,)),
            np.random.randint(-10000, 10000, size=(2,))
        )


if __name__ == '__main__':
    env = FourWayGridWorld({'use_walls': True})
    env.loc = [8, 8]
    for i in range(10000):
        env.step(np.random.uniform(size=(2,)) * 2 - 1)
    env.render()
    # _debug_plot(env.map)
    # _debug_plot(env.bool_map)
    # _test_line_intersect()
