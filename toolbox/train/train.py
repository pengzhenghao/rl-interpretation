import argparse
import copy
import pickle

import numpy as np
from ray import tune
from ray.tune.registry import register_env

from toolbox.env import register_minigrid as global_register_minigrid, \
    MiniGridWrapper
from toolbox.marl import MultiAgentEnvWrapper
from toolbox.train.deprecated_train_config import get_config
from toolbox.utils import initialize_ray


def register_bullet(env_name):
    assert isinstance(env_name, str)
    if "Bullet" in env_name:
        def make_pybullet(_=None):
            import pybullet_envs
            import gym
            print("Successfully import pybullet and found: ",
                  pybullet_envs.getList())
            return gym.make(env_name)

        register_env(env_name, make_pybullet)


def register_minigrid(env_name):
    assert isinstance(env_name, str)
    if env_name.startswith("MiniGrid"):
        global_register_minigrid()

        def make_minigrid(_=None):
            import gym_minigrid.envs
            import gym
            _ = gym_minigrid.envs
            assert "MiniGrid-Empty-16x16-v0" in [s.id for s in
                                                 gym.envs.registry.all()]
            print("Successfully import minigrid environments. We will wrap"
                  " observation using MiniGridWrapper(FlatObsWrapper).")
            return MiniGridWrapper(gym.make(env_name))

        register_env(env_name, make_minigrid)


def _get_env_name(config):
    if isinstance(config["env"], str):
        env_name = config["env"]
    elif isinstance(config["env"], dict):
        assert "grid_search" in config["env"]
        assert isinstance(config["env"]["grid_search"], list)
        assert len(config["env"]) == 1
        env_name = config["env"]["grid_search"]
    else:
        assert config["env"] is MultiAgentEnvWrapper
        env_name = config["env_config"]["env_name"]
        if isinstance(env_name, dict):
            assert "grid_search" in env_name
            assert isinstance(env_name["grid_search"], list)
            assert len(env_name) == 1
            env_name = env_name["grid_search"]
    assert isinstance(env_name, str) or isinstance(env_name, list)
    return env_name


def train(
        trainer,
        config,
        stop,
        exp_name,
        num_seeds=1,
        num_gpus=0,
        test_mode=False,
        suffix="",
        checkpoint_freq=10,
        keep_checkpoints_num=None,
        start_seed=0,
        **kwargs
):
    # initialize ray
    initialize_ray(test_mode=test_mode, local_mode=False, num_gpus=num_gpus)

    # prepare config
    used_config = {
        "seed": tune.grid_search(
            [i * 100 + start_seed for i in range(num_seeds)]),
        "log_level": "DEBUG" if test_mode else "INFO"
    }
    if config:
        used_config.update(config)
    config = copy.deepcopy(used_config)

    env_name = _get_env_name(config)

    trainer_name = trainer if isinstance(trainer, str) else trainer._name

    assert isinstance(env_name, str) or isinstance(env_name, list)
    if isinstance(env_name, str):
        env_names = [env_name]
    else:
        env_names = env_name
    for e in env_names:
        register_bullet(e)
        register_minigrid(e)

    if not isinstance(stop, dict):
        assert np.isscalar(stop)
        stop = {"timesteps_total": int(stop)}

    if keep_checkpoints_num is not None and not test_mode:
        assert isinstance(keep_checkpoints_num, int)
        kwargs["keep_checkpoints_num"] = keep_checkpoints_num
        kwargs["checkpoint_score_attr"] = "episode_reward_mean"

    if "verbose" not in kwargs:
        kwargs["verbose"] = 1 if not test_mode else 2

    # start training
    analysis = tune.run(
        trainer,
        name=exp_name,
        checkpoint_freq=checkpoint_freq if not test_mode else None,
        checkpoint_at_end=True,
        stop=stop,
        config=config,
        max_failures=20 if not test_mode else 1,
        reuse_actors=False,
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
    parser.add_argument("--num-cpus", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--num-cpus-per-worker", type=float, default=1.0)
    parser.add_argument("--num-cpus-for-driver", type=float, default=1.0)
    parser.add_argument("--env-name", type=str, default="BipedalWalker-v2")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--redis-password", type=str, default="")
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
