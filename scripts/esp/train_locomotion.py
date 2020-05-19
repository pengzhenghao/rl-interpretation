from ray import tune

from toolbox.evolution_plugin import EPTrainer, HARD_FUSE, SOFT_FUSE
from toolbox.train import train, get_train_parser

if __name__ == '__main__':
    parser = get_train_parser()
    parser.add_argument("--checkpoint-freq", "-cf", default=10, type=int)
    parser.add_argument("--adam-optimizer", action="store_true")
    args = parser.parse_args()
    exp_name = "{}-{}".format(args.exp_name, args.env_name)
    config = {
        "env": tune.grid_search([
            'Walker2DBulletEnv-v0',
            'HalfCheetahBulletEnv-v0',
            'AntBulletEnv-v0',
            'HopperBulletEnv-v0',
        ]),
        "num_sgd_iter": 10,
        "train_batch_size": 4000,
        "sample_batch_size": 200,
        "fuse_mode": tune.grid_search([SOFT_FUSE, HARD_FUSE]),
        "equal_norm": tune.grid_search([True, False]),
        "num_envs_per_worker": 8,
        "entropy_coeff": 0.001,
        "lambda": 0.95,
        "lr": 2.5e-4,
        "num_gpus": 1,  # Force to run 2 concurrently
        "evolution": {
            "train_batch_size": 4000,  # The same as PPO
            "num_workers": 10,  # default is 10,
            "num_cpus_per_worker": 0.5,
            "optimizer_type": "adam" if args.adam_optimizer else "sgd"
        },

        # locomotion config
        "kl_coeff": 1.0,
    }

    train(
        EPTrainer,
        config=config,
        stop=1e7,
        exp_name=exp_name,
        num_seeds=args.num_seeds,
        test_mode=args.test,
        checkpoint_freq=args.checkpoint_freq,
        start_seed=args.start_seed,
        verbose=1
    )
