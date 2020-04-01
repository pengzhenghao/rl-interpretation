import argparse
import pickle

import numpy as np
from ray import tune

from toolbox.marl import MultiAgentEnvWrapper
from toolbox.train.deprecated_train_config import get_config
from toolbox.utils import initialize_ray


def train(
        trainer,
        extra_config,
        stop,
        exp_name,
        num_seeds,
        num_gpus,
        test_mode=False,
        suffix="",
        checkpoint_freq=10,
keep_checkpoints_num=None,
        verbose=1,
        **kwargs
):
    # initialize ray
    initialize_ray(test_mode=test_mode, local_mode=False, num_gpus=num_gpus)

    # prepare config
    config = {
        "seed": tune.grid_search([i * 100 for i in range(num_seeds)]),
        "log_level": "DEBUG" if test_mode else "INFO"
    }
    if extra_config:
        config.update(extra_config)

    if isinstance(config["env"], str):
        env_name = config["env"]
    else:
        assert config["env"] is MultiAgentEnvWrapper
        env_name = config["env_config"]["env_name"]
    trainer_name = trainer if isinstance(trainer, str) else trainer._name

    if not isinstance(stop, dict):
        assert np.isscalar(stop)
        stop = {"timesteps_total": int(stop)}

    if keep_checkpoints_num is not None:
        assert isinstance(keep_checkpoints_num, int)
        kwargs["keep_checkpoints_num"] = keep_checkpoints_num
        kwargs["checkpoint_score_attr"] = "episode_reward_mean"

    # start training
    analysis = tune.run(
        trainer,
        name=exp_name,
        checkpoint_freq=checkpoint_freq,
        checkpoint_at_end=True,
        stop=stop,
        config=config,
        max_failures=20,
        reuse_actors=False,
        verbose=verbose,
        **kwargs
    )

    # save training progress as insurance
    pkl_path = "{}-{}-{}{}.pkl".format(exp_name, trainer_name, env_name,
                                       "" if not suffix else "-" + suffix)
    with open(pkl_path, "wb") as f:
        data = analysis.fetch_trial_dataframes()
        pickle.dump(data, f)
        print("Result is saved at: <{}>".format(pkl_path))
    return analysis


def get_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--num-cpus-per-worker", type=float, default=1.0)
    parser.add_argument("--num-cpus-for-driver", type=float, default=1.0)
    parser.add_argument("--env-name", type=str, default="BipedalWalker-v2")
    parser.add_argument("--test", action="store_true")
    return parser


if __name__ == '__main__':
    parser = get_train_parser()
    parser.add_argument("--run", type=str, default="PPO")
    args = parser.parse_args()

    print("Argument: ", args)

    run_config, stop_criterion = get_config(args.env, args.run, args.test_mode)

    train(
        args.run,
        run_config,
        stop_criterion,
        args.exp_name,
        args.num_seed,
        args.num_gpus,
        args.test_mode,
        checkpoint_freq=1
    )
