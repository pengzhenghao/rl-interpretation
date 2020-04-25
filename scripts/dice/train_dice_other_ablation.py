from ray import tune

from toolbox.dice.dice import DiCETrainer
from toolbox.dice.utils import *
from toolbox.marl import get_marl_env_config
from toolbox.train import train, get_train_parser

if __name__ == '__main__':
    parser = get_train_parser()
    parser.add_argument("--num-agents", type=int, default=5)
    parser.add_argument("--mode", type=str, required=True)
    args = parser.parse_args()

    env_name = args.env_name
    exp_name = "{}-{}".format(args.exp_name, env_name)

    large = env_name in ["Walker2d-v3", "Hopper-v3"]
    if large:
        stop = int(5e7)
    elif env_name == "Humanoid-v3":
        stop = int(2e7)
    else:
        stop = int(5e6)

    config = {
        "kl_coeff": 1.0,
        "num_sgd_iter": 10,
        "lr": 0.0001,
        'rollout_fragment_length': 200 if large else 50,
        'sgd_minibatch_size': 100 if large else 64,
        'train_batch_size': 10000 if large else 2048,
        "num_gpus": 0.4,
        "num_cpus_per_worker": args.num_cpus_per_worker,
        "num_cpus_for_driver": args.num_cpus_for_driver,
        "num_envs_per_worker": 8 if large else 5,
        'num_workers': 8 if large else 1,

        TWO_SIDE_CLIP_LOSS: tune.grid_search([True]),
        NORMALIZE_ADVANTAGE: tune.grid_search([False]),
        DELAY_UPDATE: tune.grid_search([True]),
        USE_DIVERSITY_VALUE_NETWORK: tune.grid_search([False]),
    }

    config.update(
        get_marl_env_config(env_name, tune.grid_search([args.num_agents]))
    )

    mode = args.mode
    if mode == TWO_SIDE_CLIP_LOSS:
        config[TWO_SIDE_CLIP_LOSS] = tune.grid_search([False])
    elif mode == NORMALIZE_ADVANTAGE:
        config[NORMALIZE_ADVANTAGE] = tune.grid_search([True])
    elif mode == DELAY_UPDATE:
        config[DELAY_UPDATE] = tune.grid_search([False])
    elif mode == USE_DIVERSITY_VALUE_NETWORK:
        config[USE_DIVERSITY_VALUE_NETWORK] = tune.grid_search([True])
    elif mode == "default":
        pass

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
