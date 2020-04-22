from ray import tune

from toolbox.train import train, get_train_parser

if __name__ == '__main__':
    args = get_train_parser().parse_args()
    walker_config = {
        "seed": tune.grid_search([i * 100 for i in range(3)]),
        "env": args.env_name,
        # should be fixed
        "lr": 0.0001,
        'train_batch_size': 2000,
        "num_gpus": 0.25,
        "num_cpus_per_worker": 1,
        "num_cpus_for_driver": 1,
        "num_envs_per_worker": 5,
        'num_workers': 1,
    }
    train(
        "A2C",
        exp_name=args.exp_name,
        keep_checkpoints_num=3,
        checkpoint_freq=10,
        stop=int(5e6),
        config=walker_config,
        num_gpus=args.num_gpus
    )
