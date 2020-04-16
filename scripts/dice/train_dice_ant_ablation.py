from ray import tune

from toolbox.dice import utils as constants
from toolbox.dice.dice import DiCETrainer
from toolbox.marl import get_marl_env_config
from toolbox.train import train, get_train_parser

if __name__ == '__main__':
    parser = get_train_parser()
    parser.add_argument("--num-agents", type=int, default=5)
    args = parser.parse_args()

    env_name = args.env_name
    exp_name = "{}-{}".format(args.exp_name, env_name)

    assert env_name == "Ant-v3"

    stop = int(5e6)

    config = {
        "kl_coeff": tune.grid_search([0.0, 0.5]),
        "num_sgd_iter": 10,
        "lr": tune.grid_search([0.0001, 1e-5]),
        'rollout_fragment_length': 50,
        'sgd_minibatch_size': 64,
        'train_batch_size': 2048,
        "num_gpus": 0.3,
        "num_envs_per_worker": 5,
        'num_workers': 4,
        "num_cpus_per_worker": 0.5,
        "num_cpus_for_driver": 0.5,

        constants.USE_BISECTOR: tune.grid_search([True]),
        constants.ONLY_TNB: tune.grid_search([True, False]),
    }

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
        keep_checkpoints_num=5
    )
