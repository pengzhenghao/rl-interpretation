from __future__ import absolute_import, division, print_function

import argparse

from gym.spaces import Tuple
from ray import tune
from ray.rllib.examples.twostep_game import TwoStepGame
from ray.tune import register_env

from toolbox import initialize_ray
from toolbox.marl.maddpg import MADDPGTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop", type=int, default=50000)
    parser.add_argument("--run", type=str, default="contrib/MADDPG")
    args = parser.parse_args()

    grouping = {
        "group_1": [0, 1],
    }
    obs_space = Tuple([
        TwoStepGame.observation_space,
        TwoStepGame.observation_space,
    ])
    act_space = Tuple([
        TwoStepGame.action_space,
        TwoStepGame.action_space,
    ])
    register_env(
        "grouped_twostep",
        lambda config: TwoStepGame(config).with_agent_groups(
            grouping, obs_space=obs_space, act_space=act_space))

    obs_space_dict = {
        "agent_1": TwoStepGame.observation_space,
        "agent_2": TwoStepGame.observation_space,
    }
    act_space_dict = {
        "agent_1": TwoStepGame.action_space,
        "agent_2": TwoStepGame.action_space,
    }
    config = {
        "learning_starts": 100,
        "env_config": {
            "actions_are_logits": True,
        },
        "multiagent": {
            "policies": {
                "pol1": (None, TwoStepGame.observation_space,
                         TwoStepGame.action_space, {
                             "agent_id": 0,
                         }),
                "pol2": (None, TwoStepGame.observation_space,
                         TwoStepGame.action_space, {
                             "agent_id": 1,
                         }),
            },
            "policy_mapping_fn": lambda x: "pol1" if x == 0 else "pol2",
        },
    }

    initialize_ray(test_mode=True)
    tune.run(
        # args.run,
        MADDPGTrainer,
        stop={
            "timesteps_total": args.stop,
        },
        config=dict(config, **{
            "env": TwoStepGame,
        }),
    )
