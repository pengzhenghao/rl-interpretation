import gym
import numpy as np
import pytest

from toolbox.dice.dice_sac.dice_sac_policy import DiCESACPolicy


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
    assert info["action_prob"].shape == (400, 1)
    assert np.all(info["action_prob"] >= 0.0)
    assert info["action_logp"].shape == (400, 1)
    assert info["behaviour_logits"].shape == (400, env.action_space.shape[0])

    policy._lazy_initialize({"test_my_self": policy}, None)


if __name__ == "__main__":
    pytest.main(["-v"])
