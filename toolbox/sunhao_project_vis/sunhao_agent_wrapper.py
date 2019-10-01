import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from toolbox.env.env_maker import get_env_maker

SUNHAO_AGENT_NAME = "SunhaoWrapper"


class conf(object):
    def __init__(self, hid_num=30, drop_prob=0.1):
        self.hid_num = hid_num
        self.drop_prob = drop_prob


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, config, layer_norm=True):
        super(ActorCritic, self).__init__()

        self.actor_fc1 = nn.Linear(num_inputs, 32)
        self.actor_fc2 = nn.Linear(32, config.hid_num)
        self.actor_fc3 = nn.Linear(config.hid_num, num_outputs)
        self.actor_logstd = nn.Parameter(torch.zeros(1, num_outputs))

        self.critic_fc1 = nn.Linear(num_inputs, 64)
        self.critic_fc2 = nn.Linear(64, 64)
        self.critic_fc3 = nn.Linear(64, 1)

        if layer_norm:
            self.layer_norm(self.actor_fc1, std=1.0)
            self.layer_norm(self.actor_fc2, std=1.0)
            self.layer_norm(self.actor_fc3, std=0.01)

            self.layer_norm(self.critic_fc1, std=1.0)
            self.layer_norm(self.critic_fc2, std=1.0)
            self.layer_norm(self.critic_fc3, std=1.0)

        self.config = config

    @staticmethod
    def layer_norm(layer, std=1.0, bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)

    def forward(self, states):
        """
        run policy network (actor) as well as value network (critic)
        :param states: a Tensor2 represents states
        :return: 3 Tensor2
        """
        action_mean, action_logstd = self._forward_actor(states)
        critic_value = self._forward_critic(states)
        return action_mean, action_logstd, critic_value

    def _forward_actor(self, states):
        x = torch.tanh(self.actor_fc1(states))
        x = torch.tanh(self.actor_fc2(x))
        x = F.dropout(x, p=self.config.drop_prob, training=self.training)
        action_mean = torch.tanh(self.actor_fc3(x))
        action_logstd = self.actor_logstd.expand_as(action_mean)
        return action_mean, action_logstd

    def _forward_critic(self, states):
        x = torch.tanh(self.critic_fc1(states))
        x = torch.tanh(self.critic_fc2(x))
        critic_value = self.critic_fc3(x)
        return critic_value

    def select_action(self, action_mean, action_logstd, return_logproba=True):
        """
        given mean and std, sample an action from normal(mean, std)
        also returns probability of the given chosen
        """
        action_std = torch.exp(action_logstd)
        action = torch.normal(action_mean, action_std)
        if return_logproba:
            logproba = self._normal_logproba(action, action_mean,
                                             action_logstd, action_std)
        return action, logproba

    @staticmethod
    def _normal_logproba(x, mean, logstd, std=None):
        if std is None:
            std = torch.exp(logstd)

        std_sq = std.pow(2)
        logproba = - 0.5 * math.log(2 * math.pi) - logstd - (x - mean).pow(
            2) / (2 * std_sq)
        return logproba.sum(1)

    def get_logproba(self, states, actions):
        """
        return probability of chosen the given actions under corresponding
        states of current network
        :param states: Tensor
        :param actions: Tensor
        """
        action_mean, action_logstd = self._forward_actor(states)
        action_mean = action_mean.cpu()
        action_logstd = action_logstd.cpu()
        logproba = self._normal_logproba(actions, action_mean, action_logstd)
        return logproba


class args(object):
    #     env_name = ENV_NAME
    seed = 1234
    num_episode = 800
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
    lr = 3e-4
    num_parallel_run = 1
    # tricks
    schedule_adam = 'linear'
    schedule_clip = 'linear'
    layer_norm = True
    state_norm = False
    advantage_norm = True
    lossvalue_norm = True


class FakeActionSpace(object):
    def sample(self):
        return None


class SunPolicyWrapper(object):
    def __init__(self):
        self.action_space = FakeActionSpace()


class SunAgentWrapper(object):
    _name = SUNHAO_AGENT_NAME

    def __init__(self, config=None, env=None, other=None):
        self.env = get_env_maker(env)()
        self.network = None

        config = {'ckpt': None} if config is None else config

        print("The config of sunhao agent: ", config)
        if 'ckpt' not in config:
            config['ckpt'] = None
        if config['ckpt'] is None:
            print("The given checkpoint is None.")
        else:
            self._init(config['ckpt'])
        self.config = config
        self.config['env'] = env
        self.policy = SunPolicyWrapper()

    def _init(self, ckpt):
        num_inputs = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.shape[0]
        num_units = eval(re.search(r"_(\d+)hidden", ckpt).group(1))
        self.network = ActorCritic(num_inputs, num_actions,
                                   config=conf(num_units, 0.0),
                                   layer_norm=args.layer_norm)
        self.network.load_state_dict(torch.load(ckpt))

    def compute_action(self, a_obs, prev_action=None, prev_reward=None,
                       policy_id=None, full_fetch=None):
        self.network.eval()
        action_mean, action_logstd, value = self.network((
            torch.as_tensor(
                a_obs,
                dtype=torch.float32)).unsqueeze(
            0)
        )
        action = action_mean.data.numpy()[0]
        a_info = {
            "vf_preds": 0,
        }
        return action, None, a_info

    def stop(self):
        pass

    def restore(self, ckpt):
        if self.network is None:
            self._init(ckpt)
        else:
            self.network.load_state_dict(torch.load(ckpt))
