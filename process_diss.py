import copy
import logging
import os.path as osp
import time
from math import ceil

import numpy as np
import ray
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID

from process_cluster import ClusterFinder
from rollout import rollout, replay
from utils import restore_agent, initialize_ray, _get_num_iters_from_ckpt_name

ABLATE_LAYER_NAME = "default_policy/default_model/fc2"
NO_ABLATION_UNIT_NAME = "no_ablation"
ABLATE_LAYER_NAME_DIMENSION_DICT = {
    "default_policy/default_model/fc1": 24,
    "default_policy/default_model/fc2": 256,
    "default_policy/default_model/fc_out": 256,
}

# [
# "default_policy/default_model/fc1": 24,
# "default_policy/default_model/fc2": 256,
# "default_policy/default_model/fc_out": 256,
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
        old_weight = agent.get_policy(DEFAULT_POLICY_ID).get_weights().copy()

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
        new_weight = agent.get_policy(DEFAULT_POLICY_ID).get_weights().copy()
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
        self.iter = None
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
            worker_name=None,
    ):
        self.initialized = True
        self.env_name = env_name
        self.ckpt = ckpt
        self.iter = _get_num_iters_from_ckpt_name(self.ckpt)
        self.run_name = run_name
        self.env_maker = env_maker
        self.agent_name = agent_name
        self._num_steps = None
        self.worker_name = worker_name or "Untitled Worker"
        print("{} is reset!".format(worker_name))

    @ray.method(num_return_vals=1)
    def compute_kl_divergence(self, trajectory_list):
        target_logit_list = []
        for trajectory in trajectory_list:
            actions, infos = replay(trajectory, self.agent)
            target_logit_list.append(infos["behaviour_logits"])

        # compute the KL divergence
        source_logit_list = [
            [transition[-1] for transition in trajectory]
            for trajectory in trajectory_list
        ]
        source_mean, source_log_std = np.split(
            np.concatenate(source_logit_list), 2, axis=1
        )
        target_mean, target_log_std = np.split(
            np.concatenate(target_logit_list), 2, axis=1
        )

        # Copy from ray/rllib/models/tf/tf_action_dist.py but change to Numpy
        kl_divergence = np.sum(
            target_log_std - source_log_std +
            (np.square(source_log_std) + np.square(source_mean - target_mean))
            / (2.0 * np.square(target_log_std) + 1e-9) - 0.5,
            axis=1
        )  # An array with shape (num_samples,)
        kl_divergence = np.clip(kl_divergence, 0.0, 1e38)  # to avoid inf
        averaged_kl_divergence = np.mean(kl_divergence)

        print(
            "The averaged kl divergence: {}, the total number of samples {},"
            "kl divergence shape {}".format(
                averaged_kl_divergence, len(kl_divergence), kl_divergence.shape
            )
        )
        return averaged_kl_divergence

    @ray.method(num_return_vals=1)
    def ablate(
            self,
            # num_seeds,
            num_rollouts,
            layer_name,
            unit_index,
            return_trajectory=False,
            # stack=False,
            # normalize="range",
            # log=True,
            _num_steps=None,
            save=None,
            # _extra_name=""
    ):
        assert self.initialized
        assert (save is None) or (isinstance(save, str))
        assert (isinstance(unit_index, int)) or (unit_index is None)

        self.agent = restore_agent(self.run_name, self.ckpt, self.env_name)
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
            result = rollout(
                self.agent,
                env,
                num_steps=_num_steps or 0,
                require_trajectory=True,
                require_extra_info=return_trajectory
            )
            trajectory = result['trajectory']
            episode_reward_batch.append(
                sum([transition[3] for transition in trajectory])
            )
            episode_length_batch.append(len(trajectory))
            if return_trajectory:
                behaviour_logits = result['extra_info']['behaviour_logits']
                # append the raw neural network output to trajectory
                for timestep, logit in enumerate(behaviour_logits):
                    trajectory[timestep].append(logit)
                trajectory_batch.append(trajectory)
        episode_reward_mean = np.mean(episode_reward_batch)
        episdoe_lenth_mean = np.mean(episode_length_batch)
        unit_name = _get_unit_name(layer_name, unit_index)
        result = {
            "episode_reward_mean": np.mean(episode_reward_batch),
            "episode_reward_min": min(episode_reward_batch),
            "episode_reward_max": max(episode_reward_batch),
            "episode_length_mean": np.mean(episode_length_batch),
            "episode_length_min": min(episode_length_batch),
            "episode_length_max": max(episode_length_batch),
            "num_rollouts": num_rollouts,
            "ablated_unit_index": unit_index,
            "unit_name": unit_name,
            "layer_name": layer_name,
            # This should be the original agent name
            "agent_name": self.agent_name,
            "iter": self.iter
        }
        if save:
            save_path = osp.join(save, unit_name.replace("/", "-"))
            ckpt_path = self.agent.save(save_path)
            result["checkpoint"] = ckpt_path
        if return_trajectory:
            result["trajectory"] = trajectory_batch
        print(
            "Successfully collect {} rollouts. Averaged episode reward {}."
            "Averaged episode lenth {}.".format(
                num_rollouts, episode_reward_mean, episdoe_lenth_mean
            )
        )
        self.agent.stop()
        return result


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
        num_worker=10,
        local_mode=False,
        save=None,
        _num_steps=None
):
    initialize_ray(local_mode)
    workers = [AblationWorker.remote() for _ in range(num_worker)]
    now_t_get = now_t = start_t = time.time()
    agent_count = 1
    agent_count_get = 1
    num_iteration = int(ceil(num_units / num_worker))

    result_dict = {}
    result_obj_ids = []
    kl_obj_ids = []

    # unit index None stand for the baseline test, that is no unit is ablated.
    baseline_worker = AblationWorker.remote()
    baseline_worker.reset.remote(
        run_name=run_name,
        ckpt=ckpt,
        env_name=env_name,
        env_maker=env_maker,
        agent_name=agent_name,
        worker_name="Baseline Worker"
    )

    baseline_result = copy.deepcopy(
        ray.get(
            baseline_worker.ablate.remote(
                num_rollouts=num_rollouts,
                layer_name=layer_name,
                unit_index=None,  # None stand for no ablation
                return_trajectory=True,
                _num_steps=_num_steps,
                save=save,
            )
        )
    )

    baseline_trajectory_list = copy.deepcopy(baseline_result.pop("trajectory"))
    baseline_result["kl_divergence"] = 0.0
    result_dict[_get_unit_name(layer_name, None)] = baseline_result

    unit_index_list = list(range(num_units))

    for iteration in range(num_iteration):
        start = iteration * num_worker
        end = min((iteration + 1) * num_worker, len(unit_index_list))

        for worker_index, unit_index in enumerate(unit_index_list[start:end]):
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
                unit_index=unit_index,
                save=save,
                _num_steps=_num_steps
            )
            result_obj_ids.append(obj_id)

            kl_obj_id = workers[worker_index].compute_kl_divergence.remote(
                baseline_trajectory_list
            )
            kl_obj_ids.append(kl_obj_id)

            print(
                "{}/{} (Unit {}) (+{:.1f}s/{:.1f}s) Start collecting data.".
                format(
                    agent_count,
                    len(unit_index_list),
                    unit_index,
                    time.time() - now_t,
                    time.time() - start_t,
                )
            )
            agent_count += 1
            now_t = time.time()
        for obj_id, kl_obj_id in zip(result_obj_ids, kl_obj_ids):
            result = copy.deepcopy(ray.get(obj_id))
            layer_name = result["layer_name"]
            unit_index = result["ablated_unit_index"]
            unit_name = _get_unit_name(layer_name, unit_index)
            result["kl_divergence"] = ray.get(kl_obj_id)
            result_dict[unit_name] = result
            print(
                "{}/{} (Unit {}) (+{:.1f}s/{:.1f}s) Start collecting data.".
                format(
                    agent_count_get, len(unit_index_list), unit_index,
                    time.time() - now_t_get,
                    time.time() - start_t
                )
            )
            agent_count_get += 1
            now_t_get = time.time()
        result_obj_ids.clear()
        kl_obj_ids.clear()

    ret = _parse_result_dict(result_dict)
    return ret


def generate_yaml_of_ablated_agents(
        ablation_result,
        output_path,
        run_name,
        env_name,
        ckpt=None,
):
    """
    This is an alternative function which simply parse the ablation_result
    into a yaml.
    If the ablation_result include the checkpoint information of each ablated
    agents, then we simply record those information.
    Otherwise we will build the ablated agent.
    """
    assert isinstance(ablation_result, dict)
    if not output_path.endswith(".yaml"):
        output_path = osp.join(output_path, ".yaml")

    results = []
    for unit_name, info in ablation_result.items():
        assert isinstance(info, dict)
        unit_index = info["ablated_unit_index"]
        if "checkpoint" not in info:
            assert run_name and ckpt and env_name and output_path
            layer_name = info["layer_name"]
            agent = restore_agent(run_name, ckpt, env_name)
            agent = ablate_unit(agent, layer_name, unit_index)

            dir_name = osp.dirname(output_path)
            save_path = osp.join(dir_name, unit_name.replace("/", "-"))

            ckpt_path = agent.save(save_path)
            info["checkpoint"] = ckpt_path
            info["iter"] = _get_num_iters_from_ckpt_name(ckpt_path)
            agent.stop()

        # We should update the agent name
        # Since in old name, reward is part of the name.
        # But the ablated agent has different reward.
        agent_name = info["agent_name"]
        reward = info["episode_reward_mean"]
        components = []
        for comp in agent_name.split(" "):
            if comp.startswith("rew"):
                assert comp.startswith("rew=")
                components.append("rew={:.2f}".format(reward))
            else:
                components.append(comp)
            components.append(" ")
        components.append(unit_name)
        ablated_agent_name = "".join(components)

        info["name"] = ablated_agent_name
        info["path"] = info["checkpoint"]
        info["performance"] = info["episode_reward_mean"]
        info["run_name"] = run_name
        info["env_name"] = env_name
        # info["iter"] = env_name

        assert "iter" in info

        # A strange error. The np.mean() of sth can not be saved by yaml...
        # So here is the workaround.
        result = {}
        for k, v in info.items():
            if isinstance(v, np.generic):
                v = v.item()
            result[k] = v
        results.append(result)
    results = sorted(results, key=lambda d: d["performance"])

    import yaml
    with open(output_path, 'w') as f:
        yaml.safe_dump(results, f)

    return results


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
    import pickle

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
        num_rollouts=500,
        layer_name=ABLATE_LAYER_NAME,
        num_units=ABLATE_LAYER_NAME_DIMENSION_DICT[ABLATE_LAYER_NAME],
        agent_name="PPO seed=121 rew=299.35",
        # local_mode=False,
        num_worker=24,
        save="data/ppo121_ablation_last2_layer/"
    )
    with open("ablation_result_last2_layer_0830.pkl", 'wb') as f:
        pickle.dump(result, f)

    generate_yaml_of_ablated_agents(
        result, "data/ppo121_ablation_last2_layer/result.yaml"
    )
