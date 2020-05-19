"""
This file provides the following function:

1. Allow choose PPO, SAC or TD3
2. Allow to choose two set of env/config
"""
from ray import tune

from toolbox import get_train_parser, train

if __name__ == '__main__':
    parser = get_train_parser()
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        help="The algorithm you choose. Must in [PPO, TD3, SAC]."
    )
    parser.add_argument(
        "--set",
        type=str,
        required=True,
        help="The algorithm you choose. Must in [hard, easy]."
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default="/project/BoleiZhou/pengzh/ray_results",
        help="The place to store train result."
    )
    args = parser.parse_args()
    exp_name = "{}-{}-{}".format(args.exp_name, args.algo, args.set)

    algo = args.algo

    # ===== Setup config =====
    if algo == "PPO":  # 3 CPU, 0.25 GPU
        config = {
            "kl_coeff": 1.0,
            "num_gpus": 0.25 if args.num_gpus > 0 else 0,
            "num_envs_per_worker": 5,
            'num_workers': 2,
        }
    elif algo == "SAC":  # 2 CPU, 0 GPU
        config = {
            "target_network_update_freq": 1,
            "timesteps_per_iteration": 1000,
            "learning_starts": 10000,
            "clip_actions": False,
            "normalize_actions": True,
            "evaluation_interval": 1,
            "metrics_smoothing_episodes": 5,
            "num_cpus_for_driver": 8,
            "evaluation_config": {
                "explore": False,
            }
        }
    elif algo == "TD3":  # 2 CPU, 0 GPU
        config = {
            "evaluation_interval": 5,
            "evaluation_num_episodes": 10,
            "num_cpus_for_driver": 8
        }
    else:
        raise ValueError("args.algo must in [PPO, TD3, SAC].")

    # ===== Setup the difficulty =====
    difficulty = args.set
    if difficulty == "hard":
        stop = int(5e6)
        config["env"] = tune.grid_search([
            "BipedalWalkerHardcore-v3",
            'HumanoidBulletEnv-v0',
            'HumanoidFlagrunBulletEnv-v0',
            'HumanoidFlagrunHarderBulletEnv-v0',
        ])
    elif difficulty == "easy":
        stop = int(1e6)
        config["env"] = tune.grid_search([
            "BipedalWalker-v3",
            # 'ReacherBulletEnv-v0',
            # 'PusherBulletEnv-v0',
            # 'ThrowerBulletEnv-v0',
            # 'StrikerBulletEnv-v0',
            'Walker2DBulletEnv-v0',
            'HalfCheetahBulletEnv-v0',
            'AntBulletEnv-v0',
            'HopperBulletEnv-v0',
        ])
    else:
        raise ValueError("args.set must in [hard, easy].")

    train(
        algo,
        config=config,
        stop=stop,
        exp_name=exp_name,
        num_seeds=args.num_seeds,
        test_mode=args.test,
        start_seed=args.start_seed,
        verbose=1,
        keep_checkpoints_num=10,

        # We will mainly run on CUHK cluster so we need to specify the local
        # directory to store things
        local_dir=args.local_dir
    )
