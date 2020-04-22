from ray import tune

from toolbox.train import train, get_train_parser

if __name__ == '__main__':
    args = get_train_parser().parse_args()
    walker_config = {
        "seed": tune.grid_search([i * 100 for i in range(3)]),
        "env": args.env_name,
        # should be fixed
        "kl_coeff": 1.0,
        "num_sgd_iter": 10,
        "lr": 0.0002,
        'sample_batch_size': 50,
        'sgd_minibatch_size': 64,
        'train_batch_size': 2048,
        "num_gpus": 0.25,
        "num_envs_per_worker": 5,
        'num_workers': 1,
    }
    train(
        "PPO",
        exp_name=args.exp_name,
        keep_checkpoints_num=3,
        checkpoint_freq=10,
        stop=int(5e6),
        config=walker_config,
        num_gpus=args.num_gpus
    )
