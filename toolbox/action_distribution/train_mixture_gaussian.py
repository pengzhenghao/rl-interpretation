"""Copied from toolbox/train/train.py"""
import argparse

from ray import tune
from ray.rllib.utils import merge_dicts

from toolbox.action_distribution import PPOTrainerWithoutKL, GaussianMixture, \
    register_gaussian_mixture
from toolbox.utils import initialize_ray, get_local_dir

# The arguments below is copied from
# https://github.com/araffin/rl-baselines-zoo/blob/master/hyperparams/ppo2.yml
# Notes for BipedalWalker: BipedalWalker-v2 defines "solving" as getting
# average reward of 300 over 100 consecutive trials
parser = argparse.ArgumentParser()
parser.add_argument("--exp-name", type=str, required=True)
parser.add_argument("--env", type=str, default="BipedalWalker-v2")
parser.add_argument("--run", type=str, default="PPO")
parser.add_argument("--num-seeds", type=int, default=3)
parser.add_argument("--num-gpus", type=int, default=8)
parser.add_argument("--test-mode", action="store_true")
args = parser.parse_args()

print("Argument: ", args)

target_list = []

if args.env == "BipedalWalker-v2":
    algo_specify_config_dict = {
        "PPO": {
            "config": {
                "seed": tune.grid_search(list(range(args.num_seeds))),
                # "observation_filter": "MeanStdFilter",
                "num_sgd_iter": 10,
                "num_envs_per_worker": 16,
                "gamma": 0.99,
                "entropy_coeff": 0.001,
                "lambda": 0.95,
                "lr": 2.5e-4,
            },
            "timesteps_total": 1e7
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
elif args.env == "HalfCheetah-v2":
    # https://github.com/ray-project/ray/blob/master/rllib/tuned_examples/
    # halfcheetah-ppo.yaml
    algo_specify_config_dict = {
        "PPO": {
            "config": {
                "seed": tune.grid_search((list(range(args.num_seeds)))),
                "gamma": 0.99,
                "lambda": 0.95,
                "kl_coeff": 1.0,
                'num_sgd_iter': 32,
                'lr': .0003,
                'vf_loss_coeff': 0.5,
                'clip_param': 0.2,
                'sgd_minibatch_size': 4096,
                'train_batch_size': 65536,
                "num_workers": 16,
                "num_gpus": 1,
                "grad_clip": 0.5,
                "num_envs_per_worker": 16,
                # "batch_mode": "truncate_episodes",
                # "observation_filter": "MeanStdFilter",
            },
            "stop": {
                "episode_reward_mean": 9800,
                "timesteps_total": int(1.5e8)
                # "time_total_s": 10800
            }
        }
    }
elif args.env == "Humanoid-v2":
    algo_specify_config_dict = {
        "PPO": {
            "stop": {
                "episode_reward_mean": 6000,
                "timesteps_total": int(2e8)
            },
            "config": {
                "seed": tune.grid_search(list(range(args.num_seeds))),
                "gamma": 0.995,
                "lambda": 0.95,
                "clip_param": 0.2,
                "kl_coeff": 1.0,
                "num_sgd_iter": 20,
                "lr": 0.0003,
                "horizon": 5000,
                'sgd_minibatch_size': 4096,
                'train_batch_size': 65536,
                "num_workers": 16,
                "num_envs_per_worker": 16,
                "num_gpus": 1
            }
        }
    }
elif args.env == "Walker2d-v3":
    algo_specify_config_dict = {
        "PPO": {
            "stop": {
                "timesteps_total": int(5e7),
                "episode_reward_mean": 6000
            },
            "config": {
                "seed": tune.grid_search(list(range(args.num_seeds))),
                "kl_coeff": 1.0,
                "num_sgd_iter": 20,
                "lr": 0.0001,
                "sgd_minibatch_size": 2048,
                "train_batch_size": 20000,
                "num_cpus_per_worker": 0.8,
                "num_gpus": 0.3,
                "num_cpus_for_driver": 0.5
            }
        }
    }
elif args.env == "Hopper-v2":
    algo_specify_config_dict = {
        "PPO": {
            "stop": {
                "timesteps_total": int(5e7),
                "episode_reward_mean": 4000
            },
            "config": {
                "seed": tune.grid_search(list(range(args.num_seeds))),
                "gamma": 0.995,
                "kl_coeff": 1.0,
                "num_sgd_iter": 20,
                "lr": 0.0001,
                "train_batch_size": 10000,
                "sgd_minibatch_size": 2048,
                "num_cpus_per_worker": 0.8,
                "num_gpus": 0.3,
                "num_cpus_for_driver": 0.5
            }
        }
    }
elif args.env == "atari":
    # We have result at https://github.com/ray-project/rl-experiments
    algo_specify_config_dict = {
        "PPO": {
            "stop": {
                "timesteps_total": int(2.5e7)
            },
            "config": {
                "lambda": 0.95,
                "kl_coeff": 0.5,
                "clip_rewards": True,
                "clip_param": 0.1,
                "vf_clip_param": 10.0,
                "entropy_coeff": 0.01,
                "train_batch_size": 5000,
                "sample_batch_size": 100,
                "sgd_minibatch_size": 500,
                "num_sgd_iter": 10,
                "num_workers": 10,
                "num_envs_per_worker": 5,
                "vf_share_layers": True
            }
        }
    }
else:
    raise NotImplementedError(
        "Only prepared BipedalWalker and "
        "BipedalWalkerHardcore two environments."
    )

algo_specify_config = algo_specify_config_dict[args.run]

general_config = {
    "log_level": "DEBUG" if args.test_mode else "ERROR",
    "env": args.env if args.env!= "atari" else tune.grid_search([
        "BreakoutNoFrameskip-v4",
        "BeamRiderNoFrameskip-v4",
        "QbertNoFrameskip-v4",
        "SpaceInvadersNoFrameskip-v4"
    ]),
    "num_gpus": 0.15 if 0.15<args.num_gpus else 0,
    "num_cpus_for_driver": 0.2,
    "num_cpus_per_worker": 0.75
}

run_config = merge_dicts(general_config, algo_specify_config['config'])

initialize_ray(num_gpus=args.num_gpus, test_mode=args.test_mode)

assert args.run == "PPO"
register_gaussian_mixture()

run_config["model"] = {
    "custom_action_dist": GaussianMixture.name,
    "custom_options": {
        "num_components": tune.grid_search([1, 2, 3, 5, 10])
    }
}

assert run_config['model']['custom_options']

tune.run(
    PPOTrainerWithoutKL,
    name=args.exp_name if args.exp_name != "atari" else tune.grid_search([]),
    verbose=1 if not args.test_mode else 2,
    local_dir=get_local_dir(),
    checkpoint_freq=1,
    checkpoint_at_end=True,
    max_failures=100,
    stop={"timesteps_total": algo_specify_config['timesteps_total']}
    if "timesteps_total" in algo_specify_config \
        else algo_specify_config['stop'],
    config=run_config,
)
