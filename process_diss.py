import logging
import os.path as osp
import time
from math import ceil

import copy
import numpy as np
import ray
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID

from process_cluster import ClusterFinder
from rollout import rollout, replay
from utils import restore_agent, initialize_ray

ABLATE_LAYER_NAME = "default_policy/default_model/fc_out"
NO_ABLATION_UNIT_NAME = "no_ablation"

# [
# "default_policy/default_model/fc1",
# "default_policy/default_model/fc2",
# "default_policy/default_model/fc_out"
# ]


def _get_unit_name(layer_name, unit_index):
    if unit_index is None:
        return osp.join(layer_name, NO_ABLATION_UNIT_NAME)
    assert isinstance(unit_index, int)
    return osp.join(layer_name, "unit{}".format(unit_index))


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
        # self.agent = None
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
        self.agent_name = agent_name
        self._num_steps = None
        self.worker_name = worker_name or "Untitled Worker"
        print("{} is reset!".format(worker_name))

    @ray.method(num_return_vals=1)
    def replay(self, trajectory_list_obj_id):
        trajectory_list = ray.get(trajectory_list_obj_id)
        for trajectory in trajectory_list:
            actions, infos = replay(trajectory)
            print("please stop here.")
        # compute the KL divergence



    @ray.method(num_return_vals=1)
    def ablate(
            self,
            # num_seeds,
            num_rollouts,
            layer_name,
            unit_index,
            # stack=False,
            # normalize="range",
            # log=True,
            # _num_steps=None,
            # _extra_name=""
    ):
        assert self.initialized

        # self.agent = restore_agent(self.run_name, self.ckpt, self.env_name)
        self.agent = restore_agent(self.run_name, self.ckpt, self.env_name)
        assert (isinstance(unit_index, int)) or (unit_index is None)
        if unit_index is not None:
            self.agent = ablate_unit(self.agent, layer_name, unit_index)

        trajectory_batch = []
        episode_reward_batch = []
        episode_length_batch = []
        now = start = time.time()
        env = self.env_maker()
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
            trajectory = rollout(
                self.agent, env, require_trajectory=True
            )['trajectory']
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


ABLATE_LAYER_NAME_DIMENSION_DICT = {"default_policy/default_model/fc_out": 256}


def _parse_result_dict(result_dict):
    # TODO
    return result_dict


def get_ablation_result(
        ckpt,
        run_name,
        env_name,
        env_maker,
        num_rollouts,
        layer_name,
        num_units,
        agent_name,
        num_worker=10
):

    initialize_ray()
    workers = [AblationWorker.remote() for _ in range(num_worker)]
    now_t_get = now_t = start_t = time.time()
    agent_count = 1
    agent_count_get = 1
    num_iteration = int(ceil(num_units / num_worker))

    result_dict = {}

    # run the baseline
    base_line_worker = AblationWorker.remote()
    base_line_worker.reset.remote(
        run_name=run_name,
        ckpt=ckpt,
        env_name=env_name,
        env_maker=env_maker,
        agent_name=agent_name,
        worker_name="Worker{}".format(0)
    )

    obj_id = base_line_worker.ablate.remote(
        num_rollouts=num_rollouts,
        layer_name=layer_name,
        unit_index=None
    )

    result_obj_ids = [obj_id]

    for iteration in range(num_iteration):
        start = iteration * num_worker
        end = min((iteration + 1) * num_worker, num_units)

        for worker_index, unit_index in enumerate(range(start, end)):
            workers[worker_index].reset.remote(
                run_name=run_name,
                ckpt=ckpt,
                env_name=env_name,
                env_maker=env_maker,
                agent_name=agent_name,
                worker_name="Worker{}".format(worker_index)
            )

            obj_id = workers[worker_index].ablate.remote(
                num_rollouts=num_rollouts,
                layer_name=layer_name,
                unit_index=unit_index
            )
            result_obj_ids.append(obj_id)
            print(
                "Unit {}/{} (+{:.1f}s/{:.1f}s) Start collecting data.".format(
                    unit_index,
                    num_units,
                    time.time() - now_t,
                    time.time() - start_t,
                )
            )
            agent_count += 1
            now_t = time.time()
        for obj_id in result_obj_ids:
            result = copy.deepcopy(ray.get(obj_id))
            layer_name = result["layer_name"]
            unit_index = result["ablated_unit_index"]
            unit_name = _get_unit_name(layer_name, unit_index)
            result_dict[unit_name] = result

            print(
                "Unit {}/{} (+{:.1f}s/{:.1f}s) Start collecting data.".format(
                    unit_index, num_units,
                    time.time() - now_t_get,
                    time.time() - start_t
                )
            )
            agent_count_get += 1
            now_t_get = time.time()
        result_obj_ids.clear()
    ray.get(obj_id)

    ret = _parse_result_dict(result_dict)
    return ret


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
    # from utils import initialize_ray
    #
    # initialize_ray(True)
    #
    # worker = AblationWorker.remote()
    #
    # worker.reset.remote(
    #     run_name="PPO",
    #     ckpt="~/ray_results/0810-20seeds/PPO_BipedalWalker-v2_0_seed=0_"
    #     "2019-08-10_15-21-164grca382/checkpoint_313/checkpoint-313",
    #     env_name="BipedalWalker-v2",
    #     env_maker=BipedalWalker,
    #     agent_name="TEST",
    #     worker_name="TEST_WORKER"
    # )
    #
    # obj_id = worker.ablate.remote(
    #     num_rollouts=10, layer_name=ABLATE_LAYER_NAME, unit_index=0
    # )
    # print(ray.wait([obj_id]))
    # result = ray.get(obj_id)
    # print(
    #     "Result: reward {}, length {}.".format(
    #         result['episode_reward_mean'], result['episode_length_mean']
    #     )
    # )

    def env_maker():
        env = BipedalWalker()
        env.seed(0)
        return env

    result = get_ablation_result(
        ckpt="~/ray_results/0811-0to50and100to300/"
             "PPO_BipedalWalker-v2_21_seed=121_2019-08-11_20-50-59_g4ab4_j/"
             "checkpoint_782/checkpoint-782",
        run_name="PPO",
        env_name="BipedalWalker-v2",
        env_maker=env_maker,
        num_rollouts=10,
        layer_name=ABLATE_LAYER_NAME,
        num_units=ABLATE_LAYER_NAME_DIMENSION_DICT[ABLATE_LAYER_NAME],
        agent_name="PPO seed=121 rew=299.35"
    )
