# from toolbox.modified_rllib.agent_with_mask import PPOAgentWithMask
import copy
from collections import OrderedDict

# from IPython.display import Image
import numpy as np
from ray.tune.logger import pretty_print

from toolbox import initialize_ray
from toolbox.evaluate import restore_agent_with_mask
# from toolbox.evaluate.rollout import rollout
from toolbox.evaluate.symbolic_agent import MaskSymbolicAgent


def test_agent_with_mask():
    initialize_ray(test_mode=True, local_mode=False)

    ckpt = "~/ray_results/0810-20seeds/PPO_BipedalWalker-v2_0_seed=20_2019" \
           "-08-10_16-54-37xaa2muqm/checkpoint_469/checkpoint-469"

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


def profile():
    """This function is use to profile the efficiency of agent restoring."""

    initialize_ray(num_gpus=4, test_mode=True, local_mode=True)
    ckpt = {
        'path':
        "~/ray_results/0810-20seeds/"
        "PPO_BipedalWalker-v2_0_seed=20_2019"
        "-08-10_16-54-37xaa2muqm/checkpoint_469/checkpoint-469",
        'run_name':
        "PPO",
        'env_name':
        "BipedalWalker-v2"
    }
    num_agents = 20
    master_agents = OrderedDict()
    for i in range(num_agents):
        ckpt.update(name=i)
        agent = MaskSymbolicAgent(ckpt)
        master_agents[i] = copy.deepcopy(agent)

    for i, (name, agent) in enumerate(master_agents.items()):
        print("[{}/{}] RESTORE AGENTS: NAME {}".format(i, num_agents, name))
        a = agent.get()
        print(a)


if __name__ == '__main__':
    profile()
