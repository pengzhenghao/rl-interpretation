import copy
from collections import OrderedDict

from toolbox import initialize_ray
# from toolbox.evaluate.rollout import rollout
# from toolbox.env.mujoco_wrapper import MujocoWrapper
from toolbox.evaluate.rollout import quick_rollout_from_symbolic_agents
from toolbox.evaluate.symbolic_agent import MaskSymbolicAgent
from toolbox.process_data.process_data import read_yaml

import pickle
import os.path as osp
# from toolbox.evaluate.evaluate_utils import restore_agent_with_mask

# num_agents = 2
# yaml_path = "../data/yaml/0915-halfcheetah-ppo-20-agents.yaml"
# num_rollouts = 2
# num_children = 1
# num_workers = 5
#
# normal_std = 0.1
# normal_mean = 1.0

# spawn_seed = 0
num_samples = 200  # From each agent's dataset
pca_dim = 50


def main(args):
    yaml_path = args.yaml_path
    num_agents = args.num_agents
    num_rollouts = args.num_rollouts
    num_workers = args.num_workers
    num_children = args.num_children

    normal_std = args.std
    normal_mean = args.mean

    dir_name = args.output_path

    initialize_ray(num_gpus=4, test_mode=False)

    name_ckpt_mapping = read_yaml(yaml_path, number=num_agents, mode="uniform")
    master_agents = OrderedDict()

    for name, ckpt in name_ckpt_mapping.items():
        agent = MaskSymbolicAgent(ckpt)
        master_agents[name] = agent

    spawned_agents = OrderedDict()

    for i, (name, master_agent) in \
            enumerate(master_agents.items()):

        child_name = name + " child=0"

        spawned_agents[child_name] = copy.deepcopy(
            master_agent)

        master_agent_ckpt = master_agent.agent_info['ckpt']

        for index in range(1, 1 + num_children):
            child_name = name + " child={}".format(index)
            callback_info = {
                "method": 'normal', 'mean': normal_mean,
                "std": normal_std, "seed": index + i * 100}

            spawned_agents[child_name] = \
                MaskSymbolicAgent(master_agent_ckpt, callback_info)

    rollout_ret = quick_rollout_from_symbolic_agents(
        spawned_agents, num_rollouts, num_workers,
        MujocoWrapper
    )

    for k, a in spawned_agents.items():
        a.clear()

    file_name = osp.join(
        dir_name, "{}agents_{}rollouts_{}children_{}mean_{}std.pkl".format(
            num_agents, num_rollouts, num_children, normal_mean, normal_std)
    )

    dump_obj = [rollout_ret, spawned_agents]
    with open(file_name, 'wb') as f:
        pickle.dump(dump_obj, f)
    with open(osp.join(dir_name, "args"), 'w') as f:
        f.write(args)





if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", default=1, type=int)
    args = parser.parse_args()


