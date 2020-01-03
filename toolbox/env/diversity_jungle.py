import copy

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box


def _clip(x, low, high):
    return max(min(high, x), low)


class FourWayGridWorld(gym.Env):
    def __init__(self, env_config=None):
        self.N = 17
        self.left = 10
        self.right = 10
        self.up = 10
        self.down = 10
        self.reset()
        self.observation_space = Box(0, self.N - 1, shape=(2,))
        self.action_space = Box(-1, 1, shape=(2,))

    def step(self, action):
        self.x = _clip(
            self.x + action[0],
            max(self.x - 1, 0),
            min(self.x + 1, self.N - 1)
        )
        self.y = _clip(
            self.y + action[1],
            max(self.y - 1, 0),
            min(self.y + 1, self.N - 1)
        )
        reward = self.map[int(self.x), int(self.y)]
        self.step_num += 1
        return (self.x, self.y), reward, self.step_num >= 2 * self.N, {}

    def render(self, mode=None):
        map_self = copy.deepcopy(self.map)
        map_self[int(self.loc[0]), int(self.loc[1])] = -5
        plt.imshow(map_self)

    def reset(self):
        self.map = np.ones((self.N, self.N), dtype=np.float32) * (-0.1)
        self.map[int((self.N - 1) / 2), 0] = self.left
        self.map[0, int((self.N - 1) / 2)] = self.up
        self.map[self.N - 1, int((self.N - 1) / 2)] = self.down
        self.map[int((self.N - 1) / 2), self.N - 1] = self.right
        self.x = float(np.random.randint(self.N))
        self.y = float(np.random.randint(self.N))
        self.step_num = 0
        return (self.x, self.y)

    def seed(self, s=None):
        if s is not None:
            np.random.seed(s)

# output_i = np.zeros((17, 17))
# output_j = np.zeros((17, 17))
# output_i_m = np.zeros((17, 17))
# output_j_m = np.zeros((17, 17))
# for i in range(17):
#     for j in range(17):
#         # print(np.asarray([i,j]))
#         # print(Tensor(np.asarray([i,j])))
#         # print(state,'state')
#         states = (Tensor(np.asarray([i, j])).float().unsqueeze(0).cuda())
#         # print(states,'states')
#         action_mean, action_logstd, action_w, value = network(states)
#         '''Max'''
#         rand = torch.argmax(action_w.squeeze())
#         action = action_mean[rand]
#         # print(action_mean.shape,action_w.shape)
#         # print(action_mean,action_w)
#         # action, logproba, _, prob_lst = network.select_action_by_ensemble(
#         # action_mean, action_logstd,action_w)
#         # print(action)
#         output_i[i, j] = action[0][0]
#         output_j[i, j] = action[0][1]
#
#         '''Random'''
#         rand = torch.multinomial(action_w.squeeze(), 1)
#         action = action_mean[rand]
#         # print(action.shape)
#         output_i_m[i, j] = action[0][0][0]
#         output_j_m[i, j] = action[0][0][1]
# plt.figure(figsize=(15, 10))
# plt.subplot(221)
# for i in range(17):
#     for j in range(17):
#         arrow(j, -i, output_j[i, j], -output_i[i, j], head_width=0.2,
#               shape='left')
#         # arrow(i,j,output_i[i,j],output_j[i,j],head_width=0.2,shape='left')
# # arrow(1,1,3,3,head_width=0.2,shape='left')
# xlim(-1, 17)
# ylim(-17, 1)
# yticks([2 * i - 16 for i in range(9)], [2 * i for i in
