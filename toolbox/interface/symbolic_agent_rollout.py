import copy
import logging
import os
import os.path as osp
import pickle
from collections import OrderedDict

import ray

from toolbox.evaluate.rollout import quick_rollout_from_symbolic_agents
from toolbox.evaluate.symbolic_agent import MaskSymbolicAgent
from toolbox.process_data.process_data import read_yaml

logger = logging.getLogger(__name__)


def symbolic_agent_rollout(
        yaml_path,
        num_agents,
        num_rollouts,
        num_workers,
        num_children,
        normal_std,
        normal_mean,
        dir_name,
        clear_at_end=True,
        store=True,
        mask_mode="multiply"
):
    assert ray.is_initialized()

    file_name = osp.join(
        dir_name, "{}agents_{}rollouts_{}children_{}mean_{}std.pkl".format(
            num_agents, num_rollouts, num_children, normal_mean, normal_std
        )
    )

    if os.path.exists(file_name):
        logger.warning(
            "File Detected! We will load rollout results from <{}>".format(
                file_name))
        with open(file_name, 'rb') as f:
            rollout_ret = pickle.load(f)
        return rollout_ret, file_name

    name_ckpt_mapping = read_yaml(yaml_path, number=num_agents, mode="uniform")
    master_agents = OrderedDict()

    for name, ckpt in name_ckpt_mapping.items():
        agent = MaskSymbolicAgent(ckpt, mask_mode=mask_mode)
        master_agents[name] = agent

    spawned_agents = OrderedDict()

    for i, (name, master_agent) in \
            enumerate(master_agents.items()):

        child_name = name + " child=0"

        spawned_agents[child_name] = copy.deepcopy(master_agent)

        master_agent_ckpt = master_agent.agent_info

        for index in range(1, 1 + num_children):
            child_name = name + " child={}".format(index)
            callback_info = {
                "method": 'normal',
                'mean': normal_mean,
                "std": normal_std,
                "seed": index + i * 100
            }

            spawned_agents[child_name] = \
                MaskSymbolicAgent(master_agent_ckpt, callback_info,
                                  name=child_name, mask_mode=mask_mode)

    rollout_ret = quick_rollout_from_symbolic_agents(
        spawned_agents, num_rollouts, num_workers
    )

    if clear_at_end:
        for k, a in spawned_agents.items():
            a.clear()

    os.makedirs(dir_name, exist_ok=True)

    if store:
        dump_obj = rollout_ret
        with open(file_name, 'wb') as f:
            pickle.dump(dump_obj, f)

    return rollout_ret, file_name


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml-path", required=True, type=str)
    parser.add_argument("--output-path", required=True, type=str)
    parser.add_argument("--num-agents", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--num-children", type=int, default=9)
    parser.add_argument("--num-rollouts", type=int, default=10)
    parser.add_argument("--std", type=float, required=True)
    parser.add_argument("--mean", type=float, default=1.0)
    parser.add_argument("--mask-mode", type=str, default="multiply")
    args = parser.parse_args()

    yaml_path = args.yaml_path
    num_agents = args.num_agents
    num_rollouts = args.num_rollouts
    num_workers = args.num_workers
    num_children = args.num_children

    normal_std = args.std
    normal_mean = args.mean

    dir_name = args.output_path

    from toolbox.utils import initialize_ray

    initialize_ray(test_mode=True)

    symbolic_agent_rollout(
        yaml_path, num_agents, num_rollouts, num_workers, num_children,
        normal_std, normal_mean, dir_name, mask_mode=args.mask_mode
    )
