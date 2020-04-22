from ray import tune

from ray.rllib.utils import merge_dicts

# The arguments below is copied from
# https://github.com/araffin/rl-baselines-zoo/blob/master/hyperparams/ppo2.yml
# Notes for BipedalWalker: BipedalWalker-v2 defines "solving" as getting
# average reward of 300 over 100 consecutive trials


def get_config(env, run, test):
    if env == "BipedalWalker-v2":
        algo_specify_config_dict = {
            "PPO": {
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
    elif env == "BipedalWalkerHardcore-v2":
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
    elif env == "HalfCheetah-v2":
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
                    "batch_mode": "truncate_episodes",
                    "observation_filter": "MeanStdFilter",
                },
                "stop": {
                    "episode_reward_mean": 9800,
                    "timesteps_total": int(1.5e8)
                    # "time_total_s": 10800
                }
            },
            "TD3": {
                "config": {
                    "learning_starts": 10000,
                    "pure_exploration_steps": 10000,
                    "evaluation_interval": 5,
                    "evaluation_num_episodes": 10
                },
                "stop": {
                    "episode_reward_mean": 9800,
                    "timesteps_total": int(1.5e8)
                }
            }
        }
    elif env == "Humanoid-v2":
        algo_specify_config_dict = {
            "PPO": {
                "stop": {
                    "episode_reward_mean": 6000,
                    "timesteps_total": int(2e8)
                },
                "config": {
                    "seed": tune.grid_search(list(range(10))),
                    "gamma": 0.995,
                    "lambda": 0.95,
                    "clip_param": 0.2,
                    "kl_coeff": 1.0,
                    "num_sgd_iter": 20,
                    "lr": 0.0003,
                    "horizon": 5000,
                    'sgd_minibatch_size': 4096,
                    'train_batch_size': 16384,
                    "num_workers": 16,
                    "num_envs_per_worker": 16,
                    "num_gpus": 1
                }
            }
        }
    elif env == "Walker2d-v3":
        algo_specify_config_dict = {
            "PPO": {
                "stop": {
                    "timesteps_total": int(5e7),
                    "episode_reward_mean": 4000
                },
                "config": {
                    "seed": tune.grid_search(list(range(10))),
                    "kl_coeff": 1.0,
                    "num_sgd_iter": 20,
                    "lr": 0.0001,
                    # "sgd_minibatch_size": 32768,
                    # "train_batch_size": 320000,
                    "num_cpus_per_worker": 0.8,
                    "num_gpus": 0.35,
                    "num_cpus_for_driver": 0.5
                }
            }
        }
    else:
        raise NotImplementedError(
            "Only prepared BipedalWalker and "
            "BipedalWalkerHardcore two environments."
        )

    algo_specify_config = algo_specify_config_dict[run]

    general_config = {
        "log_level": "DEBUG" if test else "ERROR",
        "env": env,
        "num_gpus": 0.15,
        "num_cpus_for_driver": 0.2,
        "num_cpus_per_worker": 0.75
    }

    run_config = merge_dicts(general_config, algo_specify_config['config'])

    if "timesteps_total" in algo_specify_config:
        stop_criterion = {
            "timesteps_total": algo_specify_config['timesteps_total']
        }
    else:
        stop_criterion = algo_specify_config['stop']

    return run_config, stop_criterion
