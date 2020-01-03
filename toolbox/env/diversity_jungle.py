import copy

import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box


# 200 points is Solved.
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
        self.observation_space = Box(0, self.N - 1, shape=(2,))
        self.action_space = Box(-1, 1, shape=(2,))

    def step(self, action):
        action = np.clip(action, -1, 1)
        new_loc = np.clip(self.loc + action, 0, self.N - 1)
        # if self.map[self.loc[0],self.loc[1]]!=0:
        self.loc = new_loc
        reward = self.map[int(self.loc[0]), int(self.loc[1])]
        self.step_num += 1
        return self.loc, reward, self.ifdone(), {}

    def ifdone(self):
        if self.step_num >= 2 * self.N:
            return True
        else:
            return False

    def render(self):
        map_self = copy.deepcopy(self.map)
        map_self[int(self.loc[0]), int(self.loc[1])] = -5
        plt.imshow(map_self)

    def reset(self):
        self.map = np.ones((self.N, self.N)) * (-0.1)
        self.map[int((self.N - 1) / 2), 0] = self.left
        self.map[0, int((self.N - 1) / 2)] = self.up
        self.map[self.N - 1, int((self.N - 1) / 2)] = self.down
        self.map[int((self.N - 1) / 2), self.N - 1] = self.right
        self.loc = np.asarray(
            [np.random.randint(self.N),
             np.random.randint(self.N)]
        )
        self.step_num = 0
        return self.loc

    def seed(self, s):
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
