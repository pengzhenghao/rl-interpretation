# from toolbox.modified_rllib.agent_with_mask import PPOAgentWithMask
import numpy as np
from ray.tune.logger import pretty_print

from toolbox.evaluate import restore_agent_with_mask
from toolbox.utils import initialize_ray


def test_agent_with_mask():
    initialize_ray(test_mode=True, local_mode=False)

    ckpt = "~/ray_results/0810-20seeds/PPO_BipedalWalker-v2_0_seed=20_2019-08-10_16-54-37xaa2muqm/checkpoint_469/checkpoint-469"

    # ckpt = None

    ret_list = []

    agent = restore_agent_with_mask("PPO", ckpt, "BipedalWalker-v2")

    # agent.compute_action(np.ones(24))

    for i in range(10):

        test_reward = agent.train()
        print(pretty_print(test_reward))
        ret_list.append(test_reward)

    print("Test end")

    agent.get_policy().set_default(
        {
            'fc_1_mask': np.ones([
                256,
            ]),
            'fc_2_mask': np.ones([
                256,
            ])
        }
    )

    for i in range(10):
        test_reward2 = agent.train()
        print(pretty_print(test_reward2))
        ret_list.append(test_reward2)

    print("Test2 end")
    return test_reward, test_reward2, ret_list

    # for i in range(10):
    #     ret = agent.train()
    #     print(pretty_print(ret))
    #     ret_list.append(ret)
    #
    # agent.get_policy().set_default({
    #     'fc_1_mask': np.ones([256, ]) * 0.5,
    #     'fc_2_mask': np.ones([256, ]) * - 0.5
    # })
    #
    # for i in range(10):
    #     ret = agent.train()
    #     print(pretty_print(ret))
    #     ret_list.append(ret)

    # agent.get_policy().set_default({
    #     'fc_1_mask': np.ones([256,]) * 0.2,
    #     'fc_2_mask': np.zeros([256,])
    # })
    #

    # print("Turn around")
    #
    # agent.get_policy().set_default({
    #     'fc_1_mask': np.ones([256, ]) * 0.0001,
    #     'fc_2_mask': np.ones([256, ]) * 0.0001
    # })
    #
    # for i in range(10):
    #     ret = agent.train()
    #     print(pretty_print(ret))
    #     ret_list.append(ret)
    #
    # return ret_list


if __name__ == '__main__':
    ret_list = test_agent_with_mask()
