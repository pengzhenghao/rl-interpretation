import argparse
import os
from collections import namedtuple
from os import makedirs as mkdir
from os.path import join as joindir

import gym
import numpy as np
import pandas as pd
import torch

from toolbox.ipd.model import ActorCritic
from toolbox.ipd.runner import ppo

parser = argparse.ArgumentParser()
parser.add_argument(
    '--hid_num', type=int, default=64, help='number of hidden unit to use'
)
parser.add_argument(
    '--drop_prob', type=float, default=0.0, help='probability of dropout'
)
parser.add_argument(
    '--env_name', type=str, default=None, help='name of environment'
)
parser.add_argument(
    '--num_episode', type=int, default=0, help='number of training episodes'
)
parser.add_argument(
    '--num_repeat',
    type=int,
    default=10,
    help='repeat the experiment for several times'
)
parser.add_argument('--use_gpu', type=int, default=0, help='if use gpu')
parser.add_argument(
    '--start_T', type=int, default=20, help='start to calculate novelty'
)
parser.add_argument(
    '--thres', type=float, default=0.0, help='threshold of novelty'
)
parser.add_argument(
    '--file_num', type=str, default=None, help='threshold of novelty'
)
parser.add_argument(
    '--weight', type=float, default=0, help='novelty reward weight'
)

config = parser.parse_args()
ENV_NAME = config.env_name
env = gym.make(ENV_NAME)
env.reset()
os.environ['CUDA_VISIBLE_DEVICES'] = str(config.use_gpu)

rwds_history = []
policy_buffer = {}
load_list = []
num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
for i in range(10):
    try:
        policy_net = ActorCritic(
            config, num_inputs, num_actions, layer_norm=True
        ).cuda()
        policy_net.load_state_dict(
            torch.load(
                ENV_NAME.split('-')[0] + config.file_num +
                '/CheckPoints/checkpoint_{'
                '0}hidden_{1}drop_prob_{'
                '2}repeat'.format(config.hid_num, config.drop_prob, i)
            )
        )
        load_list.append(
            'checkpoint_{0}hidden_{1}drop_prob_{2}repeat'.format(
                config.hid_num, config.drop_prob, i
            )
        )
        policy_buffer[str(i)] = policy_net.eval()
    except:
        pass

fn_list = []
for i in range(10):
    try:
        fn_list.append(
            np.loadtxt(
                ENV_NAME.split('-')[0] + config.file_num +
                '/Rwds/rwds_{0}hidden_{1}drop_prob_{2}repeat'.
                format(config.hid_num, config.drop_prob, i)
            )
        )
    except:
        print('wrong in ', i)

final_100_reward = []
for i in range(len(fn_list)):
    final_100_reward.append(np.mean(fn_list[i][-100:]))
print('final 100 mean reward:', np.mean(final_100_reward[:4]))

RESULT_DIR = ENV_NAME.split('-')[0] + config.file_num + '/Result_PPO'
mkdir(RESULT_DIR, exist_ok=True)
mkdir(ENV_NAME.split('-')[0] + config.file_num + '/Rwds', exist_ok=True)
mkdir(ENV_NAME.split('-')[0] + config.file_num + '/CheckPoints', exist_ok=True)

# train_list = []
#
# for i in range(10,25):
#     try:
#         policy_net = ActorCritic(num_inputs, num_actions,
#         layer_norm=True).cuda()
#         policy_net.load_state_dict(torch.load(ENV_NAME.split('-')[
#         0]+config.file_num +'/CheckPoints/EarlyStopPolicy_Suc_{0}hidden_{
#         1}threshold_{2}repeat'.format(config.hid_num,config.thres,i)))
#         load_list.append('EarlyStopPolicy_Suc_{0}hidden_{1}threshold_{
#         2}repeat'.format(config.hid_num,config.thres,i))
#         policy_buffer[str(i)] = policy_net.eval()
#     except:
#         train_list.append(i)
# print('training list now is:',train_list)

# for repeat in train_list:

#####
repeat = "DELETEME_TEST"
#####

print(len(policy_buffer), repeat)
"""
NOTICE:
    `Tensor2` means 2D-Tensor (num_samples, num_dims) 
"""

Transition = namedtuple(
    'Transition', (
        'state', 'value', 'choreo_value', 'action', 'logproba', 'mask',
        'next_state', 'reward', 'reward_novel'
    )
)


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
    lr = 3e-4
    num_parallel_run = 1
    # tricks
    schedule_adam = 'linear'
    schedule_clip = 'linear'
    layer_norm = True
    state_norm = False
    advantage_norm = True
    lossvalue_norm = True


env = gym.make(ENV_NAME)
num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
network = ActorCritic(
    config, num_inputs, num_actions, layer_norm=args.layer_norm
).cuda()
network.train()


def train(args):
    record_dfs = []
    assert len(args.num_parallel_run) == 1
    for i in range(args.num_parallel_run):
        args.seed += 1
        reward_df, rwds = ppo(args, network, policy_buffer, config)
        reward_record = pd.DataFrame(reward_df)
        reward_record['#parallel_run'] = i
        record_dfs.append(reward_record)
    record_dfs = pd.concat(record_dfs, axis=0)
    record_dfs.to_csv(
        joindir(RESULT_DIR, 'ppo-record-{}.csv'.format(args.env_name))
    )
    return rwds


# for envname in [ENV_NAME]:
#     args.env_name = envname
rwds = train(args)

torch.save(
    network.state_dict(),
    ENV_NAME.split('-')[0] + config.file_num +
    '/CheckPoints/EarlyStopPolicy_Suc_{0}hidden_{1}threshold_{2}repeat'.
    format(config.hid_num, config.thres, repeat)
)

np.savetxt(
    ENV_NAME.split('-')[0] + config.file_num +
    '/Rwds/EarlyStopPolicy_Suc_rwds_{0}hidden_{1}threshold_{2}repeat'.
    format(config.hid_num, config.thres, repeat), rwds
)

network.load_state_dict(
    torch.load(
        ENV_NAME.split('-')[0] + config.file_num +
        '/CheckPoints/EarlyStopPolicy_Suc_{0}hidden_{1}threshold_{'
        '2}repeat'.
        format(config.hid_num, config.thres,
               str(repeat) + '_temp_best')
    )
)
policy_buffer[str(repeat)] = network.eval()
