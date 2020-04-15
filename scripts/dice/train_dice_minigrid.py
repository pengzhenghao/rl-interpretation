from ray import tune

from toolbox.dice.dice import DiCETrainer
from toolbox.marl import get_marl_env_config
from toolbox.train import train, get_train_parser

if __name__ == '__main__':
    parser = get_train_parser()
    parser.add_argument("--num-agents", type=int, default=5)
    parser.add_argument("--no-delay-update", action="store_true")
    args = parser.parse_args()

    env_name = tune.grid_search([
        "MiniGrid-FourRooms-v0",
        "MiniGrid-Empty-16x16-v0",
        "MiniGrid-Empty-8x8-v0",
        "MiniGrid-DoorKey-5x5-v0"
    ])
    exp_name = "{}-{}".format(args.exp_name, env_name)

    large = False
    stop = int(2e6)

    config = {
        "kl_coeff": 1.0,
        "num_sgd_iter": 10,
        "lr": 0.0001,
        'rollout_fragment_length': 200 if large else 50,
        'sgd_minibatch_size': 100 if large else 64,
        'train_batch_size': 10000 if large else 2048,
        "num_gpus": 0.25,
        "num_cpus_per_worker": args.num_cpus_per_worker,
        "num_cpus_for_driver": args.num_cpus_for_driver,
        "num_envs_per_worker": 8 if large else 10,
        'num_workers': 8 if large else 1,
        "callbacks": {"on_train_result": None}
    }

    if env_name == "FetchPush-v1":
        stop = int(3e6)
        config.update(
            num_workers=8,
            num_envs_per_worker=10,
            gamma=0.95,
            lr=5e-4,
            delay_update=not args.no_delay_update
        )

    config.update(
        get_marl_env_config(env_name, tune.grid_search([args.num_agents]))
    )

    train(
        DiCETrainer,
        config=config,
        stop=stop,
        exp_name=exp_name,
        num_seeds=args.num_seeds,
        num_gpus=args.num_gpus,
        test_mode=args.test,
        keep_checkpoints_num=10
    )
