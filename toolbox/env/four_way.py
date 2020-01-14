import gym
import numpy as np
from gym.envs.registration import register_gym
from gym.spaces import Box

default_config = dict(
    down=10,
    up=10,
    left=10,
    right=10,
    N=16,
    early_done=False,
    int_initialize=False,
    use_walls=False,
    init_loc=None,  # None for random location
    record_trajectory=False,
    _show=True
)


# We register a non-slippery version of FrozenLake environment.
def register():
    register_gym(
        id='FourWay-v0',
        entry_point='toolbox.env.four_way:FourWayGridWorld',
        kwargs={},
    )
    register_gym(
        id='FourWayUseWalls-v0',
        entry_point='toolbox.env.four_way:FourWayGridWorld',
        kwargs={"env_config": {"use_walls": True}},
    )
    register_gym(
        id='FourWayUseWallsFixed-v0',
        entry_point='toolbox.env.four_way:FourWayGridWorld',
        kwargs={"env_config": {"use_walls": True, "init_loc": [8.0, 8.0]}},
    )
    print("Registed three environments: FourWay-v0, "
          "FourWayUseWalls-v0, FourWayUseWallsFixed-v0.")


register()


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


def _clip(x, low, high):
    """clip a scalar. This is more efficient than using numpy."""
    return max(min(x, high), low)


class FourWayGridWorld(gym.Env):
    def __init__(self, env_config=None):
        self.config = default_config
        if isinstance(env_config, dict):
            self.config.update(env_config)
        self.N = self.config['N']
        self.observation_space = Box(0, self.N, shape=(2,))
        self.action_space = Box(-1, 1, shape=(2,))
        self.map = np.ones((self.N + 1, self.N + 1), dtype=np.float32)
        self._fill_map()
        self.walls = []
        self._fill_walls()
        self.reset()

    def _fill_walls(self):
        """Let suppose you have three walls, two vertical, one horizontal."""
        if self.config['use_walls']:
            print('Building Walls!!!')
            self.walls.append(Wall([4, 6], [12, 6]))
            self.walls.append(Wall([4, 10], [12, 10]))
            self.walls.append(Wall([12, 6], [12, 10]))

    def _fill_map(self):
        self.map.fill(-0.1)
        self.map[int(self.N / 2), 0] = self.config['down']
        self.map[0, int(self.N / 2)] = self.config['left']
        self.map[self.N, int(self.N / 2)] = self.config['right']
        self.map[int(self.N / 2), self.N] = self.config['up']
        self.traj = []

    @property
    def done(self):
        if self.config['early_done']:
            return (not (0 <= self.x <= self.N - 1)) or \
                   (not (0 <= self.y <= self.N - 1)) or \
                   (self.step_num >= 2 * self.N)
        return self.step_num >= 2 * self.N

    def step(self, action):
        x = _clip(self.x + action[0], max(0, self.x - 1),
                  min(self.N, self.x + 1))
        y = _clip(self.y + action[1], max(0, self.y - 1),
                  min(self.N, self.y + 1))

        if any(w.intersect((self.x, self.y), (x, y)) for w in self.walls):
            pass
        else:
            self.x = x
            self.y = y

        loc = np.array((self.x, self.y))
        reward = self.map[int(round(self.x)), int(round(self.y))]
        self.step_num += 1
        if self.config['record_trajectory']:
            self.traj.append(loc)
        return loc, reward, self.done, {}

    def render(self, mode=None, **plt_kwargs):
        import matplotlib.pyplot as plt
        fig = plt.figure(**plt_kwargs)
        ax = fig.add_subplot()
        img = ax.imshow(
            np.transpose(self.map)[::-1, :],
            aspect=1,
            extent=[-0.5, self.N + 0.5, -0.5, self.N + 0.5],
            cmap=plt.cm.hot_r
        )
        fig.colorbar(img)
        ax.set_aspect(1)
        for w in self.walls:
            x = [w.start[0], w.end[0]]
            y = [w.start[1], w.end[1]]
            ax.plot(x, y, c='orange')
        if len(self.traj) > 1:
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
        if self.config['init_loc'] is not None:
            loc = self.set_loc(self.config['init_loc'])
        else:
            if self.config['int_initialize']:
                loc = np.random.randint(
                    0, self.N + 1, size=(2,)
                ).astype(np.float32)
            else:
                loc = np.random.uniform(
                    0, self.N, size=(2,)
                ).astype(np.float32)
        self.x, self.y = loc[0], loc[1]
        self.step_num = 0
        self.traj = [loc]
        return loc

    def seed(self, s=None):
        if s is not None:
            np.random.seed(s)

    def set_loc(self, loc):
        loc = np.asarray(loc)
        assert self.observation_space.contains(loc)
        self.x = loc[0]
        self.y = loc[1]
        return loc


def draw(compute_action, env_config=dict(), num_grids=32, **plt_kwargs):
    """compute_action is a function that take current obs (array with shape
    (2,)) as input and return the action: array with shape (2,)."""
    import matplotlib.pyplot as plt
    env_config['_show'] = False
    env = FourWayGridWorld(env_config)
    fig, ax = env.render(**plt_kwargs)
    for i in np.linspace(0, env.N, num_grids):
        for j in np.linspace(0, env.N, num_grids):
            if not env.observation_space.contains(np.array([i, j])):
                continue
            action = compute_action([i, j])
            old_loc = env.set_loc([i, j])
            new_loc, _, _, _ = env.step(action)
            diff = new_loc - old_loc
            if np.linalg.norm(diff) < 1e-9:
                continue
            ax.arrow(old_loc[0], old_loc[1], diff[0], diff[1], head_width=0.2,
                     shape='left')
    plt.show()


if __name__ == '__main__':
    test_env_config = dict(
        down=10,
        up=5,
        left=20,
        right=13,
        use_walls=True,
        init_loc=[8, 8],
        record_trajectory=True
    )
    env = FourWayGridWorld(test_env_config)
    env.loc = [8, 8]
    for i in range(1000):
        env.step(np.random.uniform(size=(2,)) * 2 - 1)
    env.render()
    compute_action = lambda _: [1, 0.5]
    draw(compute_action, test_env_config)

    # test reward
    # left
    env.reset()
    env.set_loc([0, 8])
    assert env.step([0, 0])[1] == 20

    # right
    env.reset()
    env.set_loc([16, 8])
    assert env.step([0, 0])[1] == 13

    # down
    env.reset()
    env.set_loc([8, 0])
    assert env.step([0, 0])[1] == 10

    # up
    env.reset()
    env.set_loc([8, 16])
    assert env.step([0, 0])[1] == 5

    # center
    env.reset()
    env.set_loc([8, 8])
    np.testing.assert_almost_equal(env.step([0, 0])[1], -0.1)
