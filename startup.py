"""
To Jiankai:

Please help me run the ES / A2C / DDPG / IMPALA experiments in
BipedalWalker environment.

To run this scripts, pip install gym / ray / gym[box2d], I guess it's enough.
And then run:

nohup python startup.py --exp-name 0818-es-exp --run ES > logfile.log 2>&1 &
nohup python startup.py --exp-name 0818-a2c-exp --run A2C > logfile.log 2>&1 &
nohup python startup.py --exp-name 0818-ddpg-exp --run DDPG > logfile.log 2>&1 &
nohup python startup.py --exp-name 0818-impala-exp --run IMPALA > logfile.log 2>&1 &

Doesn't matter how you run the script, name the exp or how to store the data.
I suggest not to run all scripts at the same times. Because each script will
conduct lots of trials and will definitely exhaust all resources.

Thank you very much!

-- appendix --

1. I have tried the A2C, but all 300 experiments failed??

2. The hyperparameter is copied from
https://github.com/araffin/rl-baselines-zoo/tree/master/hyperparams
I can't tell whether they are optimal, and even they have same meaning in Ray.
Just run and see.. Since the absolute performance is not so important.

Zhenghao, Aug 18
"""

import argparse
import logging

import ray
from ray import tune
from ray.rllib.utils import merge_dicts
from ray.rllib.utils.schedules import LinearSchedule

# The arguments below is copied from
# https://github.com/araffin/rl-baselines-zoo/blob/master/hyperparams/ppo2.yml
# Notes for BipedalWalker: BipedalWalker-v2 defines "solving" as getting
# average reward of 300 over 100 consecutive trials
parser = argparse.ArgumentParser()
parser.add_argument("--exp-name", type=str, required=True)
parser.add_argument("--env", type=str, default="BipedalWalker-v2")
parser.add_argument("--run", type=str, default="PPO")
args = parser.parse_args()

target_list = []

if args.env == "BipedalWalker-v2":
    algo_specify_config_dict = {
        "PPO":{
            "config": {
                "seed": tune.grid_search(list(range(20))),
                "observation_filter": "MeanStdFilter",
                "num_sgd_iter": 10,
                "num_envs_per_worker": 16,
                "gamma": 0.99,
                "entropy_coeff": 0.001,
                "lambda": 0.95,
                "lr": 2.5e-4,
            },
            "timesteps_total": 1e6
        },
        "ES": {
            "config": {
                "seed": tune.grid_search(list(range(50)))
            },
            "timesteps_total": int(1e9),
        },
        "A2C": {
            "config": {
                "num_workers": 1,
                "num_envs_per_worker": 16,
                "entropy_coeff": 0.0,
                "sample_batch_size": 256,
                "train_batch_size": 500,
                "seed": tune.grid_search(list(range(300)))
            },
            "timesteps_total": int(5e6),
        },
        "DDPG": {
            "config": {
                "target_noise": 0.287,
                "sample_batch_size": 256,
                "actor_lr": 0.000527,
                "gamma": 0.999,
                "buffer_size": 100000,
                "seed": tune.grid_search(list(range(300)))
            },
            "timesteps_total": int(1e6)
        },
        "IMPALA": {
            "config": {
                "num_workers": 1,
                "num_envs_per_worker": 16,
                "seed": tune.grid_search(list(range(300)))
            },
            "timesteps_total": int(5e6),
        },
    }
elif args.env == "BipedalWalkerHardcore-v2":
    algo_specify_config_dict = {
        "ES": {
            "config": {
                "seed": tune.grid_search(list(range(10)))
            },
            "timesteps_total": int(1e9),
        },
        "A2C": {
            "config": {
                "num_workers": 1,
                "num_envs_per_worker": 16,
                "entropy_coeff": 0.001,
                "sample_batch_size": 256,
                "seed": tune.grid_search(list(range(20)))
            },
            "timesteps_total": int(10e7),
        },
        "PPO": {
            "config": {
                "num_envs_per_worker": 16,
                "gamma": 0.99,
                "entropy_coeff": 0.001,
                "num_sgd_iter": 10,
                "lambda": 0.95,
                "lr_schedule": [[0, 2.5e-4], [8e7, 1e-5]],
                "seed": tune.grid_search(list(range(100)))
            },
            "timesteps_total": int(10e7)
        }
    }
else:
    raise NotImplementedError(
        "Only prepared BipedalWalker and "
        "BipedalWalkerHardcore two environments."
    )

algo_specify_config = algo_specify_config_dict[args.run]

general_config = {
    "env": args.env,
    "num_gpus": 0.15,
    "num_cpus_for_driver": 0.2
}

run_config = merge_dicts(general_config, algo_specify_config['config'])

ray.init(logging_level=logging.ERROR, log_to_driver=False)
tune.run(
    args.run,
    name=args.exp_name,
    verbose=1,
    checkpoint_freq=1,
    checkpoint_at_end=True,
    stop={"timesteps_total": algo_specify_config['timesteps_total']},
    config=run_config,
)
