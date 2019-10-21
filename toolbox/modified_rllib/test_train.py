# from toolbox.modified_rllib.agent_with_mask import PPOAgentWithMask
from toolbox.evaluate import restore_agent_with_mask
from toolbox.utils import initialize_ray

import numpy as np

def test_agent_with_mask():
    initialize_ray(test_mode=True, local_mode=False)

    agent = restore_agent_with_mask("PPO", None, "BipedalWalker-v2")

    agent.compute_action(np.ones(24))

    agent.get_policy().set_default({
        'fc_1_mask': np.ones([256,]),
        'fc_2_mask': np.zeros([256,])
    })

    ret = agent.train()

    return ret


if __name__ == '__main__':
    ret = test_agent_with_mask()
