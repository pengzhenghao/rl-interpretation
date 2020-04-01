from ray import tune

from toolbox.dies.dies import DiESTrainer
from toolbox.marl import get_marl_env_config
from toolbox.train import train, get_train_parser

if __name__ == '__main__':
    parser = get_train_parser()
    parser.add_argument("--num-agents", type=int, default=10)
    args = parser.parse_args()

    env_name = args.env_name
    exp_name = "{}-{}".format(args.exp_name, env_name)
    stop = int(5e7)

    config = {
        "num_sgd_iter": 10,
        "num_envs_per_worker": 1,
        "entropy_coeff": 0.001,
        "lambda": 0.95,
        "lr": 2.5e-4,

        # 'sample_batch_size': 200 if large else 50,
        # 'sgd_minibatch_size': 100 if large else 64,
        # 'train_batch_size': 10000 if large else 2048,
        "num_gpus": 1,
        "num_cpus_per_worker": 1,
        "num_cpus_for_driver": 2,
        'num_workers': 16
    }

    config.update(get_marl_env_config(
        env_name, tune.grid_search([args.num_agents])))

    train(
        DiESTrainer,
        extra_config=config,
        stop=stop,
        exp_name=exp_name,
        num_seeds=args.num_seeds,
        num_gpus=args.num_gpus,
        test_mode=args.test
    )
