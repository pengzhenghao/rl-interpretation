import shutil
import tempfile

import gym
import numpy as np
import pytest

from toolbox import train, initialize_ray
from toolbox.dice.dice_sac.dice_sac import DiCESACTrainer
from toolbox.dice.dice_sac.dice_sac_policy import DiCESACPolicy
from toolbox.marl import get_marl_env_config, MultiAgentEnvWrapper


@pytest.fixture()
def dice_sac_policy():
    env_name = "BipedalWalker-v2"
    env = gym.make(env_name)
    policy = DiCESACPolicy(env.observation_space, env.action_space, {
        "env": env_name
    })
    return env, policy


def test_policy(dice_sac_policy):
    env, policy = dice_sac_policy

    act, _, info = policy.compute_actions(np.random.random([400, 24]))
    assert act.shape == (400, 4)
    assert info["action_prob"].shape[0] == 400
    assert np.all(info["action_prob"] >= 0.0)
    assert info["action_logp"].shape[0] == 400
    assert info["behaviour_logits"].shape == (400, env.action_space.shape[0])

    policy._lazy_initialize({"test_my_self": policy}, None)


@pytest.fixture()
def dice_sac_trainer():
    initialize_ray(test_mode=True, local_mode=False)
    env_name = "BipedalWalker-v2"
    num_agents = 3
    env = gym.make(env_name)
    trainer = DiCESACTrainer(
        get_marl_env_config(env_name, num_agents, normalize_actions=True),
        env=MultiAgentEnvWrapper)
    return env, trainer


def test_trainer(dice_sac_trainer):
    env, trainer = dice_sac_trainer
    train_result = trainer.train()


def regression_test():
    num_agents = 3
    local_dir = tempfile.mkdtemp()
    initialize_ray(test_mode=True, local_mode=False)
    train(DiCESACTrainer,
          {
              "gamma": 0.95,
              "target_network_update_freq": 32,
              "tau": 1.0,
              "train_batch_size": 32,
              "optimization": {
                  "actor_learning_rate": 0.005,
                  "critic_learning_rate": 0.005,
                  "entropy_learning_rate": 0.0001
              },
              **get_marl_env_config(
                  "CartPole-v0", num_agents, normalize_actions=True
              )
          },
          {"episode_reward_mean": 150 * num_agents}, exp_name="DELETEME",
          local_dir=local_dir, test_mode=True)
    shutil.rmtree(local_dir, ignore_errors=True)


if __name__ == "__main__":
    # pytest.main(["-v"])
    regression_test()
