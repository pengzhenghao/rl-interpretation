"""
This file train a PPO policy.

Usage:
    python train_PPO.py \
    --exp-name=EXPERIMENT_NAME \
    --num_workers=1 \
    --num_envs_per_worker=1 \
    --num_cpus_per_worker=1 \
    --sample_batch_size=100 \
    --train_batch_size=1000 \
    --num_iters=100 \
    -rf=REWARD_FUNCTION_NAME \
    --revival \
    --scene merge \
    --seed -1 \
    -g 8 \

Take a look at /scripts/XXXX.sh for example using cluster to train RL agent.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path as osp
import random
import shutil
import uuid

import numpy as np
import ray
import tensorflow as tf
from ray import tune
from ray.rllib.models import Model
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

import gym
# from env import RaySimulator
# from env.constants import *
# from env.utils import get_config_path
# from tf_utils import conv1d, fc, conv_to_fc

parser = argparse.ArgumentParser()

# configure ray
parser.add_argument("--num_envs_per_worker", '-e', type=int, default=1)
parser.add_argument("--num_workers", "-w", type=int, default=8)
parser.add_argument("--num_cpus_per_worker", "-c", type=int, default=1)
parser.add_argument("--num_gpus", "-g", type=int, default=8)
parser.add_argument("--restore", type=str)
parser.add_argument("--seed", type=int, default=1997)
parser.add_argument("--exp-name", dest='exp_name', type=str, default="tmp")

# configure environment
# parser.add_argument("--num_agents", '-a', type=int, default=16)
# parser.add_argument("--revival", action="store_true", default=False)
# parser.add_argument("--random-environment", action="store_true", default=False)
parser.add_argument("--local-mode", action="store_true", default=False)
# parser.add_argument(
#     "--reward-function-name", "-rf", type=str, default='default'
# )
# parser.add_argument("--scene", type=str, required=True)
# parser.add_argument("--not-allow-goal", action="store_true", default=False)

# configure learning
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--num-policies", type=int, default=1)
parser.add_argument("--sample_batch_size", "-s", type=int, default=500)
parser.add_argument("--train_batch_size", "-t", type=int, default=4000)
parser.add_argument("--num_sgd_iter", type=int, default=10)
parser.add_argument("--num_iters", "-i", type=int, default=100)

# class CnnPolicy(Model):
#     def _build_layers_v2(self, input_dict, num_outputs, options):
#         self.obs_in = input_dict["obs"]
#         conv_X, fc_X = tf.split(
#             self.obs_in, [72, int(self.obs_in.shape[1]) - 72], 1
#         )
#         conv_X_pad = tf.concat([conv_X, conv_X[:, 0:4]], 1)
#         with tf.variable_scope(tf.VariableScope(tf.AUTO_REUSE,
#                                                 "shared"), reuse=tf.AUTO_REUSE,
#                                auxiliary_name_scope=False):
#             activ = tf.tanh
#             h1 = activ(
#                 conv1d(
#                     conv_X_pad,
#                     'pi_conv1',
#                     nf=128,
#                     rf=4,
#                     stride=2,
#                     init_scale=np.sqrt(2)
#                 )
#             )
#             h2 = activ(
#                 conv1d(
#                     h1,
#                     'pi_conv2',
#                     nf=8,
#                     rf=8,
#                     stride=2,
#                     init_scale=np.sqrt(2)
#                 )
#             )
#             h3 = conv_to_fc(
#                 tf.concat(
#                     [
#                         tf.reshape(
#                             h2, [-1, h2.shape[1] * (h2.shape[2] // 4), 4]
#                         ), fc_X
#                     ], 1
#                 )
#             )
#             h4 = activ(fc(h3, 'pi_fc1', nh=16, init_scale=np.sqrt(2)))
#             pi = fc(h4, 'pi', num_outputs, init_scale=0.01)
#
#             h1 = activ(
#                 conv1d(
#                     conv_X_pad,
#                     'vf_conv1',
#                     nf=16,
#                     rf=4,
#                     stride=2,
#                     init_scale=np.sqrt(2)
#                 )
#             )
#             h2 = activ(
#                 conv1d(
#                     h1,
#                     'vf_conv2',
#                     nf=4,
#                     rf=3,
#                     stride=2,
#                     init_scale=np.sqrt(2)
#                 )
#             )
#             h3 = conv_to_fc(
#                 tf.concat([h2, fc_X], 1)
#             )  # [18,4] & [7/5,4], concat axis 1
#             h4 = activ(fc(h3, 'vf_fc1', nh=16, init_scale=np.sqrt(2)))
#         return pi, h4


def on_ep_end(info):
    ep = info['episode']

    agent_info = ep._agent_to_last_info
    histories = ep._agent_reward_history

    d = {
        k: 0
        for k in (
            "col_g_rate", "col_b_rate", "col_a_rate", "collide_agent_rate",
            "collide_boundary_rate", "collide_gate_rate", "success_rate",
            "pass_rate", "die_of_large_angle"
        )
    }

    for k, v in agent_info.items():

        state = v['state']
        agent_ep_len = len(histories[k])
        assert agent_ep_len != 0

        if state == GATE_COLLISION:
            d['col_g_rate'] += 1 / agent_ep_len
            d['collide_gate_rate'] += 1
        elif state == VEHICLE_COLLISION:
            d['col_a_rate'] += 1 / agent_ep_len
            d['collide_agent_rate'] += 1
        elif state == BOUNDARY_COLLISION:
            d['col_b_rate'] += 1 / agent_ep_len
            d['collide_boundary_rate'] += 1
        elif state == WRONG_GATE:
            d['pass_rate'] += 1
        elif state == PASSED:
            d['success_rate'] += 1
            d['pass_rate'] += 1
        elif state == DIE_OF_LARGE_ANGLE:
            d["die_of_large_angle"] += 1

    for k, v in d.items():
        ep.custom_metrics[k] = v / len(agent_info)


if __name__ == "__main__":
    args = parser.parse_args()
    tmp_dir = osp.abspath(
        "/tmp/ray_{}".format(args.exp_name if args.exp_name else uuid.uuid4())
    )
    shutil.rmtree(tmp_dir, ignore_errors=True)
    ray.init(temp_dir=tmp_dir, local_mode=args.local_mode)

    if args.seed == -1:
        args.seed = np.random.randint(0, 10000)
        print("[INFO] We will use {} as random seed!".format(args.seed))

    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    env = gym.make("CartPole-v0")

    def build_env(env_config=None):
        e = RaySimulator(
            args.num_agents,
            config_path=get_config_path(args.scene),
            stack=4,
            revival=args.revival,
            random_env=args.random_environment,
            reward_function_name=args.reward_function_name,
            extra_config={"allow_goal": False} if args.not_allow_goal else None
        )  # ray 0.7
        seed = env_config.worker_index * 10 + args.seed \
            if env_config else args.seed
        e.seed(seed)
        return e

    register_env("tollgate", build_env)
    # ModelCatalog.register_custom_model("CnnPolicy", CnnPolicy)
    single_env = build_env()
    obs_space = single_env.observation_space[0]
    act_space = single_env.action_space[0]

    print(
        "Current observation space: {}\nCurrent action space: {}".format(
            obs_space, act_space
        )
    )

    policy_graphs = {
        "policy": (
            None,  # ray 0.7
            obs_space,
            act_space,
            {
                "model": {
                    "custom_model": "CnnPolicy",
                },
                "vf_share_layers": True
            }
        )
    }

    if args.restore:
        restore = args.restore
        assert isinstance(restore, str)
        print("prepare to restore: ", args.restore)
    else:
        restore = None

    tune.run(
        "PPO",
        name=args.exp_name,
        checkpoint_at_end=True,
        checkpoint_freq=10,
        restore=restore,
        stop={"training_iteration": args.num_iters},
        config={
            "env": "tollgate",
            "log_level": "WARNING",
            "observation_filter": "MeanStdFilter",

            # Resource param.
            "num_workers": args.num_workers,
            "num_envs_per_worker": args.num_envs_per_worker,
            "num_cpus_per_worker": args.num_cpus_per_worker,
            "sample_batch_size": args.sample_batch_size,
            "train_batch_size": args.train_batch_size,
            "num_gpus": args.num_gpus,

            # PPO param.
            'lr': args.lr,
            'lr_schedule': [(0, args.lr), (1e7, 1e-4)],
            "entropy_coeff": 0.01,
            "num_sgd_iter": args.num_sgd_iter,
            "callbacks": {
                "on_episode_end": tune.function(on_ep_end)
            },
            "multiagent": {
                "policy_graphs": policy_graphs,
                "policy_mapping_fn": tune.function(lambda agent_id: "policy"),
            },
        },
    )
