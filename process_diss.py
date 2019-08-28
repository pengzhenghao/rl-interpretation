import os.path as osp

import numpy as np
from process_cluster import ClusterFinder

import ray
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID

import pandas

ABLATE_LAYER_NAMES = [
    # "default_policy/default_model/fc1",
    # "default_policy/default_model/fc2",
    "default_policy/default_model/fc_out"
]


def ablate_unit(agent, layer_name, index, _test=False):
    # It should be noted that the agent is MODIFIED in-place in this function!

    # get weight dict
    policy = agent.get_policy(DEFAULT_POLICY_ID)

    if _test:
        old_weight = agent.get_policy(DEFAULT_POLICY_ID).get_weigths().copy()

    weight_dict = policy._variables.get_weights()
    assert isinstance(weight_dict, dict)

    # get the target matrix's name
    weight_name = osp.join(layer_name, "kernel")
    assert weight_name in weight_dict
    matrix = weight_dict[weight_name]
    assert matrix.ndim == 2

    # ablate
    matrix[index, :] = 0
    weight_dict[weight_name] = matrix
    ablated_weight = weight_dict

    # set back the ablated matrix
    policy = agent.get_policy(DEFAULT_POLICY_ID)
    policy._variables.set_weights(ablated_weight)

    if _test:
        new_weight = agent.get_policy(DEFAULT_POLICY_ID).get_weigths().copy()
        assert not np.all(old_weight==new_weight)

    return agent



@ray.remote
class DissectWorker(object):
    def __init__(self):
        pass

    @ray.method(num_return_vals=0)
    def reset(
            self,
            run_name,
            ckpt,
            env_name,
            env_maker,
            agent_name,
            padding=None,
            padding_length=None,
            padding_value=None,
            worker_name=None,
    ):
        pass

    @ray.method(num_return_vals=2)
    def dissect(self):
        pass
        return None, None


def parse_representation_dict(representation_dict, *args, **kwargs):
    cluster_df = pandas.DataFrame.from_dict(representation_dict).T
    return cluster_df


def get_diss_representation(
        name_ckpt_mapping, run_name, env_name, env_maker, num_seeds,
        num_rollouts, *args, **kwargs
):
    # Input: a batch of agent, Output: a batch of representation
    pass


def get_dissect_cluster_finder():
    cluster_df = None
    cf = ClusterFinder(cluster_df)
    return cf


if __name__ == '__main__':
    # test codes here.
    from ray.rllib.agents.ppo import PPOAgent
    from utils import initialize_ray
    initialize_ray()
    a = PPOAgent(env="BipedalWalker-v2")
    ow = a.get_policy(DEFAULT_POLICY_ID).get_weights().copy()
    na = ablate_unit(a, ABLATE_LAYER_NAMES[0], 10)
    nw = na.get_policy(DEFAULT_POLICY_ID).get_weights().copy()
