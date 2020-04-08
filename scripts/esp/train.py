from ray import tune

from toolbox.evolution_plugin import EPTrainer, HARD_FUSE, SOFT_FUSE
from toolbox.train import train, get_train_parser

if __name__ == '__main__':
    args = get_train_parser().parse_args()
    exp_name = "{}-{}".format(args.exp_name, args.env_name)
    config = {
        "env": args.env_name,
        "num_sgd_iter": 10,
        "train_batch_size": 4000,
        "sample_batch_size": 200,
        "fuse_mode": tune.grid_search([SOFT_FUSE, HARD_FUSE]),
        "num_envs_per_worker": 8,
        "entropy_coeff": 0.001,
        "lambda": 0.95,
        "lr": 2.5e-4,
        "num_gpus": 1,  # Force to run 2 concurrently
        "evolution": {
            "train_batch_size": 4000,  # The same as PPO
            "num_workers": 10,  # default is 10,
            "optimizer_type": "adam",
            "num_cpus_per_worker": 0.5,
        }
    }
    train(
        EPTrainer,
        extra_config=config,
        stop=1e7,
        exp_name=exp_name,
        num_seeds=args.num_seeds,
        num_gpus=args.num_gpus,
        test_mode=args.test,
        keep_checkpoints_num=5,
        start_seed=args.start_seed
    )
