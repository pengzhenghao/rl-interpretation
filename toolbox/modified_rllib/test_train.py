# from toolbox.modified_rllib.agent_with_mask import PPOAgentWithMask
from toolbox.evaluate import restore_agent_with_mask
from toolbox.utils import initialize_ray


def test_agent_with_mask():
    initialize_ray(test_mode=True)

    agent = restore_agent_with_mask("PPO", None, "BipedalWalker-v2")

    ret = agent.train()

    return ret


if __name__ == '__main__':
    ret = test_agent_with_mask()
