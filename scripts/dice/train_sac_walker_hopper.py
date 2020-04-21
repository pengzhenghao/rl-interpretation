from ray import tune

from toolbox.train import train, get_train_parser

if __name__ == '__main__':
    args = get_train_parser().parse_args()

    exp_name = args.exp_name  # It's "12230-ppo..." previously....
    env_name = args.env_name
    stop = int(2e6)
    num_gpus = args.num_gpus

    sac_config = {
        "env": tune.grid_search([
            # "HalfCheetah-v3",
            "Walker2d-v3",
            # "Ant-v3",
            "Hopper-v3",
            # "Humanoid-v3"
        ]),

        "horizon": 1000,

        "sample_batch_size": 1,
        "train_batch_size": 256,

        "target_network_update_freq": 1,
        "timesteps_per_iteration": 1000,
        "learning_starts": 10000,

        "clip_actions": False,
        "normalize_actions": True,

        "evaluation_interval": 1,
        "metrics_smoothing_episodes": 5,
        "num_cpus_for_driver": 2,
    }

    train(
        "SAC",
        exp_name=exp_name,
        keep_checkpoints_num=5,
        stop=stop,
        config=sac_config,
        num_gpus=args.num_gpus,
        num_seeds=args.num_seeds
    )
