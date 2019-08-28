import logging
import os.path as osp
import time

import numpy as np
import pandas
import ray
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID

from process_cluster import ClusterFinder
from rollout import rollout
from utils import restore_agent

ABLATE_LAYER_NAME = "default_policy/default_model/fc_out"

# [
# "default_policy/default_model/fc1",
# "default_policy/default_model/fc2",
# "default_policy/default_model/fc_out"
# ]


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
    assert index < matrix.shape[0]
    matrix[index, :] = 0
    weight_dict[weight_name] = matrix
    ablated_weight = weight_dict

    # set back the ablated matrix
    policy = agent.get_policy(DEFAULT_POLICY_ID)
    policy._variables.set_weights(ablated_weight)

    if _test:
        new_weight = agent.get_policy(DEFAULT_POLICY_ID).get_weigths().copy()
        assert not np.all(old_weight == new_weight)

    return agent


@ray.remote
class AblationWorker(object):
    """
    This worker only conduct the ablation of ONE unit of a given agent!
    """

    def __init__(self):
        self._num_steps = None
        self.agent = None
        self.agent_name = None
        self.env_maker = None
        self.worker_name = "Untitled Worker"
        self.initialized = False
        self.run_name = None
        self.ckpt = None
        self.env_name = None
        # self.postprocess_func = lambda x: x
        # self.padding_value = None

    @ray.method(num_return_vals=0)
    def reset(
            self,
            run_name,
            ckpt,
            env_name,
            env_maker,
            agent_name,
            # unit_index,
            worker_name=None,
    ):
        self.initialized = True
        self.env_name = env_name
        self.ckpt = ckpt
        self.run_name = run_name
        self.env_maker = env_maker
        self.env = env_maker()
        self.agent_name = agent_name
        self._num_steps = None
        self.worker_name = worker_name or "Untitled Worker"
        print("{} is reset!".format(worker_name))

    @ray.method(num_return_vals=1)
    def ablate(
            self,
            # num_seeds,
            num_rollouts,
            unit_index,
            # stack=False,
            # normalize="range",
            # log=True,
            # _num_steps=None,
            # _extra_name=""
    ):
        assert self.initialized

        layer_name = ABLATE_LAYER_NAME

        self.agent = restore_agent(self.run_name, self.ckpt, self.env_name)
        ablated_agent = restore_agent(self.run_name, self.ckpt, self.env_name)
        assert isinstance(unit_index, int)
        ablated_agent = ablate_unit(
            ablated_agent, layer_name, unit_index
        )

        trajectory_batch = []
        episode_reward_batch = []
        episode_length_batch = []
        now = start = time.time()
        for rollout_index in range(num_rollouts):
            # print some running information
            logging.info(
                "({}) Agent {}, Rollout [{}/{}], Time [+{}s/{}s]".format(
                    self.worker_name, self.agent_name, rollout_index,
                    num_rollouts,
                    time.time() - now,
                    time.time() - start
                )
            )
            now = time.time()

            # collect trajectory
            trajectory = \
            rollout(ablated_agent, self.env, require_trajectory=True)[
                'trajectory']
            trajectory_batch.append(trajectory)
            episode_reward_batch.append(
                sum([transition[3] for transition in trajectory])
            )
            episode_length_batch.append(len(trajectory))

        episode_reward_mean = np.mean(episode_reward_batch)
        episdoe_lenth_mean = np.mean(episode_length_batch)
        result = {
            "trajectory": trajectory_batch,
            "episode_reward_mean": np.mean(episode_reward_batch),
            "episode_reward_min": min(episode_reward_batch),
            "episode_reward_max": max(episode_reward_batch),
            "episode_length_mean": np.mean(episode_length_batch),
            "episode_length_min": min(episode_length_batch),
            "episode_length_max": max(episode_length_batch),
            "num_rollouts": num_rollouts,
            "ablated_unit_index": unit_index,
            "layer_name": layer_name,
            "agent_name": self.agent_name
            # "KL_divergence": None,  # some metric for similarity
        }
        print(
            "Successfully collect {} rollouts. Averaged episode reward {}."
            "Averaged episode lenth {}.".format(
                num_rollouts, episode_reward_mean, episdoe_lenth_mean
            )
        )
        return result


def parse_representation_dict(representation_dict, *args, **kwargs):
    raise NotImplementedError


def get_diss_representation(
        name_ckpt_mapping, run_name, env_name, env_maker, num_seeds,
        num_rollouts, *args, **kwargs
):
    # Input: a batch of agent, Output: a batch of representation
    raise NotImplementedError


def get_dissect_cluster_finder():
    cluster_df = None
    cf = ClusterFinder(cluster_df)
    raise NotImplementedError


if __name__ == '__main__':
    # test codes here.
    from gym.envs.box2d import BipedalWalker
    from utils import initialize_ray

    initialize_ray(True)

    worker = AblationWorker.remote()

    worker.reset.remote(
        run_name="PPO",
        ckpt="~/ray_results/0810-20seeds/PPO_BipedalWalker-v2_0_seed=0_"
             "2019-08-10_15-21-164grca382/checkpoint_313/checkpoint-313",
        env_name="BipedalWalker-v2",
        env_maker=BipedalWalker,
        agent_name="TEST",
        worker_name="TEST_WORKER"
    )

    obj_id = worker.ablate.remote(num_rollouts=10, unit_index=0)
    print(ray.wait([obj_id]))
    result = ray.get(obj_id)
    print("Result: reward {}, length {}.".format(
        result['episode_reward_mean'], result['episode_length_mean'])
    )
