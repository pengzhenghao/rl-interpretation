import copy
from itertools import combinations

from numpy import *


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
        plt.imshow(map_self)

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


class config:
    def __init__(self, hid_num=64, env_name='Walker2d-v3', num_episode=10,
                 num_repeat=1, choices_n=1, use_gpu=5, thres=0.22, coef=0.01):
        self.hid_num = hid_num
        self.env_name = env_name
        self.num_episode = num_episode
        self.num_repeat = num_repeat
        self.use_gpu = use_gpu
        self.choices_n = choices_n
        self.thres = thres
        self.coef = coef
        self.experiment_name = '{0}_{1}_{2}_DivInit'.format(self.choices_n,
                                                            self.thres,
                                                            self.coef)


config = config()

os.environ['CUDA_VISIBLE_DEVICES'] = str(config.use_gpu)
SELECTION_METHOD = 'param'
# SELECTION_METHOD = 'ddpg_critic'

ENV_NAME = 'FourWayGridWorld-vself'

USE_GPU = True if 0 <= config.use_gpu <= 7 else False
env = FourWayGridWorld(17)
env.reset()

actor_proba_logger = []

rwds_history = []
for repeat in range(config.num_repeat):
    Eval_reward_logger = []
    import torch
    import torch.nn as nn
    import torch.optim as opt
    from torch import Tensor
    from collections import namedtuple
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    from os.path import join as joindir
    from os import makedirs as mkdir
    import pandas as pd
    import numpy as np
    import math

    Transition = namedtuple('Transition', (
        'state', 'value', 'action', 'logproba', 'mask', 'next_state', 'reward',
        'action_next'))
    EPS = 1e-10
    RESULT_DIR = 'Result_' + config.experiment_name
    mkdir(RESULT_DIR, exist_ok=True)
    mkdir(ENV_NAME.split('-')[0] + '/CheckPoints_' + config.experiment_name,
          exist_ok=True)
    mkdir(ENV_NAME.split('-')[0] + '/Rwds_' + config.experiment_name,
          exist_ok=True)
    rwds = []


    class args(object):
        env_name = ENV_NAME
        seed = 1234
        num_episode = config.num_episode
        batch_size = 2048
        max_step_per_round = 2000
        gamma = 0.995
        lamda = 0.97
        log_num_episode = 1
        num_epoch = 10
        minibatch_size = 256
        clip = 0.2
        loss_coeff_value = 0.5
        loss_coeff_entropy = 0.01
        loss_coeff_div = config.coef
        lr = 3e-4
        num_parallel_run = 1
        # tricks
        schedule_adam = 'linear'
        schedule_clip = 'linear'
        layer_norm = True
        state_norm = False
        advantage_norm = True
        lossvalue_norm = True


    class RunningStat(object):
        def __init__(self, shape):
            self._n = 0
            self._M = np.zeros(shape)
            self._S = np.zeros(shape)

        def push(self, x):
            x = np.asarray(x)
            assert x.shape == self._M.shape
            self._n += 1
            if self._n == 1:
                self._M[...] = x
            else:
                oldM = self._M.copy()
                self._M[...] = oldM + (x - oldM) / self._n
                self._S[...] = self._S + (x - oldM) * (x - self._M)

        @property
        def n(self):
            return self._n

        @property
        def mean(self):
            return self._M

        @property
        def var(self):
            return self._S / (self._n - 1) if self._n > 1 else np.square(
                self._M)

        @property
        def std(self):
            return np.sqrt(self.var)

        @property
        def shape(self):
            return self._M.shape


    class ZFilter:
        """
        y = (x-mean)/std
        using running estimates of mean,std
        """

        def __init__(self, shape, demean=True, destd=True, clip=10.0):
            self.demean = demean
            self.destd = destd
            self.clip = clip

            self.rs = RunningStat(shape)

        def __call__(self, x, update=True):
            if update: self.rs.push(x)
            if self.demean:
                x = x - self.rs.mean
            if self.destd:
                x = x / (self.rs.std + 1e-8)
            if self.clip:
                x = np.clip(x, -self.clip, self.clip)
            return x

        def output_shape(self, input_space):
            return input_space.shape


    class ActorCritic(nn.Module):
        def __init__(self, num_inputs, num_outputs, choices_n=config.choices_n,
                     layer_norm=True):
            super(ActorCritic, self).__init__()
            self.choices_n = choices_n
            self.actor_fc1 = nn.Linear(num_inputs, 32)
            self.actor_fc2 = nn.Linear(32, config.hid_num)

            self.actor_fcw1 = nn.Linear(num_inputs, 32)
            self.actor_fcw2 = nn.Linear(32, config.hid_num)

            self.actor_fc3 = nn.ModuleList()
            self.actor_w = nn.ModuleList()
            self.actor_logstd = nn.Parameter(
                torch.zeros(choices_n, 1, num_outputs))
            self.actor_w_old = nn.Parameter(torch.zeros(choices_n, 1, 1))
            for i in range(choices_n):
                self.actor_fc3.append(nn.Linear(config.hid_num, num_outputs))
                self.actor_w.append(nn.Linear(config.hid_num, 1))
            self.sm0 = nn.Softmax(dim=0)
            self.sm1 = nn.Softmax(dim=1)
            self.sm2 = nn.Softmax(dim=2)
            self.critic_fc1 = nn.Linear(num_inputs, 64)
            self.critic_fc2 = nn.Linear(64, 64)
            self.critic_fc3 = nn.Linear(64, 1)

            self.critic_ddpg_fc1 = nn.Linear(num_inputs + num_outputs, 64)
            self.critic_ddpg_fc2 = nn.Linear(64, 64)
            self.critic_ddpg_fc3 = nn.Linear(64, 1)

            if layer_norm:
                self.layer_norm(self.actor_fc1, std=1.0)
                self.layer_norm(self.actor_fc2, std=1.0)
                self.layer_norm(self.actor_fcw1, std=1.0)
                self.layer_norm(self.actor_fcw2, std=1.0)
                for i in range(choices_n):
                    self.layer_norm(self.actor_fc3[i], std=1.0)
                    self.layer_norm(self.actor_w[i], std=1.0)

                self.layer_norm(self.critic_fc1, std=1.0)
                self.layer_norm(self.critic_fc2, std=1.0)
                self.layer_norm(self.critic_fc3, std=1.0)

        @staticmethod
        def layer_norm(layer, std=1.0, bias_const=0.0):
            torch.nn.init.orthogonal_(layer.weight, std)
            torch.nn.init.constant_(layer.bias, bias_const)

        def forward(self, states):
            action_mean, action_logstd, action_w = self._forward_actor(states)
            critic_value = self._forward_critic(states)
            return action_mean, action_logstd, action_w, critic_value

        def _forward_actor(self, states):
            x = torch.tanh(self.actor_fc1(states))
            x = torch.tanh(self.actor_fc2(x))

            xw = torch.tanh(self.actor_fcw1(states))
            xw = torch.tanh(self.actor_fcw2(xw))
            action_mean = []
            # action_logstd = []
            action_w = []
            for i in range(self.choices_n):
                action_mean.append(torch.tanh(self.actor_fc3[i](x)))
                action_w.append(self.actor_w[i](xw))

            action_mean = torch.stack(action_mean)
            action_w = torch.stack(action_w)
            action_w = self.sm0(action_w)

            action_w_old = self.sm0(self.actor_w_old)

            action_logstd = self.actor_logstd.expand(action_mean.size())
            # print(action_mean.shape)
            # print('sssss',action_w_old,action_w)
            # action_w = action_w.expand((action_mean.size(0),
            # action_mean.size(1),1))
            return action_mean, action_logstd, action_w

        def _forward_ddpg_critic(self, states, action):
            x = torch.tanh(
                self.critic_ddpg_fc1(torch.cat([states, action], 1)))
            x = torch.tanh(self.critic_ddpg_fc2(x))
            ddpg_q_value = torch.tanh(self.critic_ddpg_fc3(x))
            return ddpg_q_value

        def _forward_critic(self, states):
            x = torch.tanh(self.critic_fc1(states))
            x = torch.tanh(self.critic_fc2(x))
            critic_value = self.critic_fc3(x)
            return critic_value


        def select_action_by_ensemble(self, action_mean, action_logstd,
                                      action_w, return_logproba=True):
            prob_list = action_w.view(-1)
            # print(prob_list.shape)
            rand = torch.argmax(prob_list)
            # rand = np.random.choice(self.choices_n, p = prob_list)
            action_mean_selected = action_mean[rand]
            # print(action_mean_selected.shape) #[1,1,6]
            action_logstd_selected = action_logstd[rand]  # [1,1,6]
            action_std_selected = torch.exp(action_logstd_selected)
            action_selected = torch.normal(action_mean_selected,
                                           action_std_selected)
            # print(action_selected.shape) #[1,1,6]
            if return_logproba:
                logproba = self._normal_logproba_MoG(action_selected,
                                                     action_mean,
                                                     action_logstd,
                                                     prob_list.unsqueeze(-1))
            return action_selected, logproba, rand, prob_list

        def select_action_by_parameter(self, action_mean, action_logstd,
                                       action_w, return_logproba=True):
            # print(action_w.shape,action_mean.shape,action_logstd.shape)
            # print(action_w)

            prob_list = action_w.view(-1)
            # print(prob_list.shape)
            rand = torch.multinomial(prob_list, 1)
            # rand = np.random.choice(self.choices_n, p = prob_list)
            action_mean_selected = action_mean[rand]
            # print(action_mean_selected.shape) #[1,1,6]
            action_logstd_selected = action_logstd[rand]  # [1,1,6]
            action_std_selected = torch.exp(action_logstd_selected)
            action_selected = torch.normal(action_mean_selected,
                                           action_std_selected)
            # print(action_selected.shape) #[1,1,6]
            if return_logproba:
                logproba = self._normal_logproba_MoG(action_selected,
                                                     action_mean,
                                                     action_logstd,
                                                     prob_list.unsqueeze(-1))
            return action_selected, logproba, rand, prob_list

        def _normal_logproba_MoG(self, x, mean, logstd, prob_list):
            std = torch.exp(logstd)

            # embed()
            exp_part = (torch.exp(-(x.expand(self.choices_n, mean.size(1),
                                             mean.size(2)) - mean).pow(2) / (
                                          2 * std.pow(2))) / std)
            # print(x.shape,mean.shape,logstd.shape,prob_list.shape,
            # exp_part.shape,prob_list.unsqueeze(-1).shape)#,
            # prob_list.expand_as(exp_part).shape)
            proba = prob_list.unsqueeze(-1).expand_as(exp_part) / (
                    (2 * math.pi) ** (0.5)) * exp_part
            proba = proba.sum(0)
            logproba = torch.log(proba)
            return logproba.sum(1)

        def get_logproba(self, states, actions):
            # print(states.shape)
            action_mean, action_logstd, action_w = self._forward_actor(states)
            # embed()
            if SELECTION_METHOD == 'ddpg_critic':
                self._forward_ddpg_critic(states, action_mean)
                action, logproba, _, prob_lst = self.select_action_by_ddpg(
                    action_mean, action_logstd, states)
            elif SELECTION_METHOD == 'param':
                prob_lst = action_w.squeeze(2)
            # print(prob_lst.shape)
            # embed()
            # logproba = []
            # print(actions.permute(1,0,2).shape, action_mean.shape,
            # action_logstd.shape,prob_lst.shape)
            # for i in range(len(actions)):
            #    logproba.append(self._normal_logproba_MoG(actions.permute(
            #    1,0,2), action_mean, action_logstd,prob_lst))
            logproba = self._normal_logproba_MoG(actions.permute(1, 0, 2),
                                                 action_mean, action_logstd,
                                                 prob_lst)
            # print('logprobashape',(logproba).shape)
            return logproba


    class Memory(object):
        def __init__(self):
            self.memory = []

        def push(self, *args):
            self.memory.append(Transition(*args))

        def sample(self):
            return Transition(*zip(*self.memory))

        def __len__(self):
            return len(self.memory)


    env = FourWayGridWorld(17)
    num_inputs = 2
    num_actions = 2

    if USE_GPU:
        network = ActorCritic(num_inputs, num_actions,
                              choices_n=config.choices_n,
                              layer_norm=args.layer_norm).cuda()
        print('using GPU-{}'.format(config.use_gpu))
    else:
        network = ActorCritic(num_inputs, num_actions,
                              layer_norm=args.layer_norm)
    network.train()


    def ppo(args):

        env = FourWayGridWorld(17)
        num_inputs = 2
        num_actions = 2

        optimizer = opt.Adam(network.parameters(), lr=args.lr)

        running_state = ZFilter((num_inputs,), clip=5.0)

        # record average 1-round cumulative reward in every episode
        reward_record = []
        global_steps = 0

        lr_now = args.lr
        clip_now = args.clip

        for i_episode in range(args.num_episode):
            # step1: perform current policy to collect trajectories
            # this is an on-policy method!
            memory = Memory()
            num_steps = 0
            reward_list = []
            len_list = []
            while num_steps < args.batch_size:
                state = env.reset()
                if args.state_norm:
                    state = running_state(state)
                reward_sum = 0
                for t in range(args.max_step_per_round):
                    if USE_GPU:
                        action_mean, action_logstd, action_w, value = network(
                            Tensor(state).float().unsqueeze(0).cuda())
                    else:
                        print('NotImplementedError')
                        # action_mean, action_logstd, value = network(
                        # Tensor(state).float().unsqueeze(0))

                    if SELECTION_METHOD == 'ddpg_critic':
                        action, logproba, _, prob_lst = \
                            network.select_action_by_ddpg(
                            action_mean, action_logstd,
                            Tensor(state).float().unsqueeze(0).cuda())
                    elif SELECTION_METHOD == 'param':
                        action, logproba, _, prob_lst = \
                            network.select_action_by_parameter(
                            action_mean, action_logstd, action_w)
                    else:
                        print("Error Method")

                    action = action.cpu().data.numpy()[0]
                    logproba = logproba.cpu().data.numpy()[0]
                    # rand_channel = rand_channel.cpu().data.numpy()[0]
                    # print(action)
                    next_state, reward, done = env.step(action[0])

                    reward_sum += reward
                    if args.state_norm:
                        next_state = running_state(next_state)
                    mask = 0 if done else 1

                    if USE_GPU:
                        a_m_next, a_l_next, w_next, v_next = network(
                            Tensor(next_state).float().unsqueeze(0).cuda())
                    else:
                        print('NotImplemented Error')
                    if SELECTION_METHOD == 'ddpg_critic':
                        action_next, logproba_next, _next, _prob_next = \
                            network.select_action_by_ddpg(
                            a_m_next, a_l_next,
                            Tensor(next_state).float().unsqueeze(0).cuda())
                        action_next = action_next.cpu().data.numpy()[0]
                    elif SELECTION_METHOD == 'param':
                        action_next, logproba_next, _next, _prob_next = \
                            network.select_action_by_parameter(
                            a_m_next, a_l_next, w_next)
                        action_next = action_next.cpu().data.numpy()[0]
                    else:
                        print("Error Method")
                    memory.push(state, value, action, logproba, mask,
                                next_state, reward, action_next)

                    if done:
                        break

                    state = next_state

                num_steps += (t + 1)
                global_steps += (t + 1)
                reward_list.append(reward_sum)
                len_list.append(t + 1)
            reward_record.append({
                'episode': i_episode,
                'steps': global_steps,
                'meanepreward': np.mean(reward_list),
                'meaneplen': np.mean(len_list)})
            rwds.extend(reward_list)
            batch = memory.sample()
            batch_size = len(memory)

            # step2: extract variables from trajectories
            rewards = Tensor(batch.reward).float()
            values = Tensor(batch.value).float()
            masks = Tensor(batch.mask).float()
            actions = Tensor(batch.action).float()
            states = Tensor(batch.state).float()
            oldlogproba = Tensor(batch.logproba).float()
            # rand_channel = Tensor(batch.rand_channel).float()

            action_nexts = Tensor(batch.action_next).float()

            next_states = Tensor(batch.next_state).float()

            returns = Tensor(batch_size).float()
            deltas = Tensor(batch_size).float()
            advantages = Tensor(batch_size).float()

            prev_return = 0.
            prev_value = 0.
            prev_advantage = 0.
            for i in reversed(range(batch_size)):
                returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
                deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - \
                            values[i]
                # ref: https://arxiv.org/pdf/1506.02438.pdf (generalization
                # advantage estimate)
                advantages[i] = deltas[
                                    i] + args.gamma * args.lamda * \
                                prev_advantage * \
                                masks[i]

                prev_return = returns[i]
                prev_value = values[i]
                prev_advantage = advantages[i]
            if args.advantage_norm:
                advantages = (advantages - advantages.mean()) / (
                        advantages.std() + EPS)

            for i_epoch in range(
                    int(args.num_epoch * batch_size / args.minibatch_size)):
                '''update ppo'''
                if USE_GPU:
                    minibatch_ind = np.random.choice(batch_size,
                                                     args.minibatch_size,
                                                     replace=False)
                    minibatch_states = states[minibatch_ind]

                    minibatch_next_states = next_states[minibatch_ind]
                    minibatch_rewards = rewards[minibatch_ind]
                    minibatch_masks = masks[minibatch_ind]
                    minibatch_action_nexts = action_nexts[minibatch_ind]

                    minibatch_actions = actions[minibatch_ind]
                    # minibatch_rand = rand_channel[minibatch_ind]
                    minibatch_oldlogproba = oldlogproba[minibatch_ind]
                    minibatch_newlogproba = network.get_logproba(
                        minibatch_states.cuda(),
                        minibatch_actions.cuda()).cpu()
                    minibatch_advantages = advantages[minibatch_ind]
                    minibatch_returns = returns[minibatch_ind]
                    minibatch_newvalues = network._forward_critic(
                        minibatch_states.cuda()).cpu().flatten()

                    ratio = torch.exp(
                        minibatch_newlogproba - minibatch_oldlogproba)
                    surr1 = ratio * minibatch_advantages
                    surr2 = ratio.clamp(1 - clip_now,
                                        1 + clip_now) * minibatch_advantages
                    loss_surr_cpu = - torch.mean(torch.min(surr1, surr2))

                    '''divergence_loss_regularization'''
                    if config.choices_n > 1:
                        minibatch_action_mean, minibatch_action_std, \
                        minibatch_action_w = network._forward_actor(
                            minibatch_states.cuda())
                        # print(minibatch_action_mean.shape)
                        (i_1, i_2) = list(
                            combinations([i for i in range(config.choices_n)],
                                         2))[0]
                        mean_distance = torch.mean(torch.norm((
                                minibatch_action_mean[
                                    i_1] -
                                minibatch_action_mean[
                                    i_2]),
                            dim=1))
                        for (i_1, i_2) in list(combinations(
                                [i for i in range(config.choices_n)], 2))[1:]:
                            mean_distance = torch.min(mean_distance,
                                                      torch.mean(torch.norm((
                                                              minibatch_action_mean[
                                                                  i_1] -
                                                              minibatch_action_mean[
                                                                  i_2]),
                                                          dim=1)))
                    else:
                        mean_distance = torch.as_tensor(999)

                    if args.lossvalue_norm:
                        minibatch_return_6std = 6 * minibatch_returns.std()
                        loss_value_cpu = torch.mean(
                            (minibatch_newvalues - minibatch_returns).pow(
                                2)) / minibatch_return_6std
                    else:
                        loss_value_cpu = torch.mean(
                            (minibatch_newvalues - minibatch_returns).pow(2))

                    loss_entropy_cpu = torch.mean(torch.exp(
                        minibatch_newlogproba) * minibatch_newlogproba)
                    if mean_distance <= 2 * config.thres:
                        total_loss_cpu = loss_surr_cpu + \
                                         args.loss_coeff_value * \
                                         loss_value_cpu + \
                                         args.loss_coeff_entropy * \
                                         loss_entropy_cpu - \
                                         args.loss_coeff_div * \
                                         mean_distance.cpu()
                    else:
                        total_loss_cpu = loss_surr_cpu + \
                                         args.loss_coeff_value * \
                                         loss_value_cpu + \
                                         args.loss_coeff_entropy * \
                                         loss_entropy_cpu
                    optimizer.zero_grad()
                    total_loss_cpu.backward()
                    optimizer.step()

                    if SELECTION_METHOD == "ddpg_critic":
                        '''update ddpg_critic'''
                        # print('minibatch_next_states',
                        # minibatch_next_states.shape)
                        # Compute the target Q value
                        # pred_action_mean, pred_action_std =
                        # network._forward_actor(minibatch_next_states.cuda())

                        target_Q = network._forward_ddpg_critic(
                            minibatch_next_states.cuda(),
                            minibatch_action_nexts.cuda()).cpu().detach()
                        target_Q = (minibatch_rewards + (
                                minibatch_masks * args.gamma *
                                target_Q)).detach()

                        # Get current Q estimate
                        current_Q = network._forward_ddpg_critic(
                            minibatch_states.cuda(),
                            minibatch_actions.cuda()).cpu()

                        # Compute critic loss
                        ddpg_critic_loss = F.mse_loss(current_Q, target_Q)

                        # Optimize the critic
                        optimizer.zero_grad()
                        ddpg_critic_loss.backward()
                        optimizer.step()

            if args.schedule_clip == 'linear':
                ep_ratio = 1 - (i_episode / args.num_episode)
                clip_now = args.clip * ep_ratio

            if args.schedule_adam == 'linear':
                ep_ratio = 1 - (i_episode / args.num_episode)
                lr_now = args.lr * ep_ratio
                # set learning rate
                # ref: https://stackoverflow.com/questions/48324152/
                for g in optimizer.param_groups:
                    g['lr'] = lr_now

            if i_episode % args.log_num_episode == 0:
                # print('Finished episode: {} Reward: {:.4f} total_loss = {
                # :.4f} = {:.4f} + {} * {:.4f} + {} * {:.4f}' \
                #    .format(i_episode, reward_record[-1]['meanepreward'],
                #    total_loss.data, loss_surr.data, args.loss_coeff_value,
                #    loss_value.data, args.loss_coeff_entropy,
                #    loss_entropy.data))
                print("total loss cc", i_episode,
                      reward_record[-1]['meanepreward'], total_loss_cpu.data,
                      loss_surr_cpu.data, loss_value_cpu.data,
                      loss_entropy_cpu.data, mean_distance.data)
                done = False
                running_state = ZFilter((num_inputs,), clip=5.0)
                i_episode = 0
                for ep in range(1):
                    i_episode += 1
                    state = env.reset()
                    if args.state_norm:
                        state = running_state(state)
                    eval_reward_sum = 0
                    for t in range(999):  # args.max_step_per_round):
                        network.eval()
                        # action_mean, action_logstd, value= network((
                        # torch.as_tensor(state,
                        # dtype=torch.float32)).unsqueeze(0))
                        action_mean, action_logstd, action_w, value = network(
                            Tensor(state).float().unsqueeze(0).cuda())

                        '''Max'''
                        action, logproba, _, prob_lst = \
                            network.select_action_by_ensemble(
                            action_mean, action_logstd, action_w)
                        #                 '''Random'''
                        #                 action, logproba, _, prob_lst =
                        #                 network.select_action_by_parameter(action_mean, action_logstd,action_w)
                        #                         '''choice'''
                        #                         action, logproba, _,
                        #                         prob_lst =
                        #                         network.select_action_by_number(action_mean, action_logstd,action_w,Rnd_num=choice_i)

                        action = action.cpu().data.numpy()[0]
                        next_state, reward, done = env.step(action)
                        eval_reward_sum += reward

                        if done:
                            # print(t)
                            # print('reward_sum',reward_sum)
                            break
                            # done = False
                        state = next_state
                    print(t)
                    print('reward_sum', eval_reward_sum,
                          reward_record[-1]['meanepreward'],
                          eval_reward_sum - reward_record[-1]['meanepreward'])
                    Eval_reward_logger.append(
                        [eval_reward_sum, reward_record[-1]['meanepreward'],
                         eval_reward_sum - reward_record[-1]['meanepreward']])
                print('-----------------')

        return reward_record


    def test(args):
        record_dfs = []
        for i in range(args.num_parallel_run):
            args.seed += 1
            reward_record = pd.DataFrame(ppo(args))
            reward_record['#parallel_run'] = i
            record_dfs.append(reward_record)
        record_dfs = pd.concat(record_dfs, axis=0)
        record_dfs.to_csv(
            joindir(RESULT_DIR, 'ppo-record-{}.csv'.format(args.env_name)))


    if __name__ == '__main__':
        for envname in [ENV_NAME]:
            args.env_name = envname
            test(args)

    torch.save(network.state_dict(), ENV_NAME.split('-')[
        0] + '/CheckPoints_{0}/checkpoint_{1}hidden_{2}repeat'.format(
        config.experiment_name, config.hid_num, repeat))
    np.savetxt(
        ENV_NAME.split('-')[0] + '/Rwds_{0}/rwds_{1}hidden_{2}repeat'.format(
            config.experiment_name, config.hid_num, repeat), rwds)
    np.savetxt(ENV_NAME.split('-')[
                   0] + '/Rwds_{0}/Eval_rwds_{1}hidden_{2}repeat'.format(
        config.experiment_name, config.hid_num, repeat), Eval_reward_logger)
