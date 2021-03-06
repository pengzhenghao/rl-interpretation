from ray import tune

from toolbox.dice.dice_sac import DiCESACTrainer
from toolbox.dice.dice_sac.dice_sac_config import constants
from toolbox.marl import get_marl_env_config
from toolbox.train import train, get_train_parser

if __name__ == '__main__':
    parser = get_train_parser()
    parser.add_argument("--num-agents", type=int, default=5)
    args = parser.parse_args()

    exp_name = args.exp_name
    stop = int(1.5e6)

    config = {
        "env": tune.grid_search([
            # "HalfCheetah-v3",
            # "Walker2d-v3",
            "Ant-v3",  # <<== This script run only one env for validation
            # "Hopper-v3",
            # "Humanoid-v3"
        ]),

        "num_cpus_for_driver": 2,

        constants.DELAY_UPDATE: tune.grid_search([True, False]),

        # SAC config
        "horizon": 1000,
        "sample_batch_size": 1,
        "train_batch_size": 256,
        "target_network_update_freq": 1,
        "timesteps_per_iteration": 1000,
        "learning_starts": 10000,
        "clip_actions": True,
        # "normalize_actions": True,  <<== This is handled by MARL env
        "evaluation_interval": 1,
        "metrics_smoothing_episodes": 5,
    }
    config.update(get_marl_env_config(
        config["env"], args.num_agents, normalize_actions=True
    ))

    config["evaluation_config"] = dict(
        # env_config=config["env_config"],
        explore=False
    )

    train(
        DiCESACTrainer,
        config=config,
        stop=stop,
        exp_name=exp_name,
        num_seeds=args.num_seeds,
        num_gpus=args.num_gpus,
        test_mode=args.test,
        keep_checkpoints_num=5
    )
