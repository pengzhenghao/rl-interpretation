from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

from toolbox.action_distribution import GaussianMixture
from toolbox.atv import ANA2CTrainer, ANA3CTrainer, ANIMPALATrainer
from toolbox.evolution.modified_ars import GaussianARSTrainer
from toolbox.evolution.modified_es import GaussianESTrainer
from toolbox.train import train, get_train_parser


def get_dynamic_trainer(algo):
    if algo == "PPO":
        base = PPOTrainer
    elif algo == "ES":
        base = GaussianESTrainer
    elif algo == "A2C":
        base = ANA2CTrainer
    elif algo == "A3C":
        base = ANA3CTrainer
    elif algo == "IMPALA":
        base = ANIMPALATrainer
    elif algo == "ARS":
        base = GaussianARSTrainer
    else:
        raise NotImplementedError()
    return base


if __name__ == '__main__':
    parser = get_train_parser()
    parser.add_argument("--algo", type=str, required=True)
    args = parser.parse_args()

    algo = args.algo
    test = args.test
    exp_name = "{}-{}-{}-{}seeds".format(args.exp_name, algo,
                                         args.env_name, args.num_seeds)

    algo_specify_config = {
        "PPO": {
            "num_sgd_iter": 10,
            "num_envs_per_worker": 8,
            "entropy_coeff": 0.001,
            "lambda": 0.95,
            "lr": 2.5e-4,
        },
        "ES": {"model": {"vf_share_layers": False}},
        "ARS": {"model": {"vf_share_layers": False}},
        "A2C": {
            # "num_envs_per_worker": 8,
            "model": {"vf_share_layers": False},
            "entropy_coeff": 0.0,
            # "lr": 1e-5
        },
        "A3C": {
            # "num_envs_per_worker": 8,
            # "sample_batch_size": 20,
            # "entropy_coeff": 0.001,
            "entropy_coeff": 0.0,
            # "lr": 1e-5,
            "model": {"vf_share_layers": False}
        },
        "IMPALA": {
            "num_envs_per_worker": 8,
            "entropy_coeff": 0.001,
            "lr": 1e-4,
            "model": {"vf_share_layers": False}
        },
    }

    algo_specify_stop = {
        "PPO": 1e7,
        "ES": 1e9,
        "ARS": 1e9,
        "A2C": 5e7,
        "A3C": 5e7,
        "IMPALA": 5e7
    }

    stop = int(algo_specify_stop[algo])
    config = algo_specify_config[algo]

    # Update model config (Remove all model config above)
    config["model"] = {
        "vf_share_layers": False,
        "custom_action_dist": GaussianMixture.name,
        "custom_options": {
            "num_components": tune.grid_search([2, 3, 5])
        }
    }

    config["env"] = args.env_name

    if algo in ["ES", "ARS"]:
        config["num_gpus"] = 0
        config["num_cpus_per_worker"] = 0.5
        config["num_workers"] = 20

    train(
        get_dynamic_trainer(algo),
        extra_config=config,
        stop=stop,
        exp_name=exp_name,
        num_seeds=args.num_seeds,
        num_gpus=args.num_gpus,
        test_mode=test,
        keep_checkpoints_num=5
    )
