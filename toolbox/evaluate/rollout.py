"""
This filed is copied from ray.rllib.rollout but with little modification.
"""
# !/usr/bin/env python

from __future__ import absolute_import, division, print_function

import argparse
import collections
import copy
import logging
import os
import pickle
import time
from math import ceil

import numpy as np
import ray
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.evaluation.episode import _flatten_action
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID

from toolbox.evaluate.evaluate_utils import (
    restore_agent, restore_agent_with_activation
)
from toolbox.evaluate.tf_model import PPOTFPolicyWithActivation
from toolbox.process_data.process_data import read_yaml
from toolbox.utils import initialize_ray, ENV_MAKER_LOOKUP, has_gpu

EXAMPLE_USAGE = """
Example Usage via RLlib CLI:
    rllib rollout /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl

Example Usage via executable:
    ./rollout.py /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl
"""
LOG_INTERVAL_STEPS = 500

ENV_NAME_PERIOD_FEATURE_LOOKUP = {
    "BipedalWalker-v2": [7, 12],
    "HalfCheetah-v2": [2, 11, 15]
}


class DefaultMapping(collections.defaultdict):
    """default_factory now takes as an argument the missing key."""

    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value


def default_policy_agent_mapping(unused_agent_id):
    return DEFAULT_POLICY_ID


def get_dataset_path(ckpt, num_rollout, seed):
    if ckpt is None:
        ckpt = "/tmp/tmp"
    ckpt = os.path.abspath(os.path.expanduser(ckpt))
    return "{}_rollout{}_seed{}.pkl".format(ckpt, num_rollout, seed)


class RolloutWorkerWrapper(object):
    @classmethod
    def as_remote(cls, num_cpus=None, num_gpus=None, resources=None):
        return ray.remote(
            num_cpus=num_cpus, num_gpus=num_gpus, resources=resources
        )(cls)

    def __init__(self, force_rewrite=False):
        self.initialized = False
        self.already_reset = False
        self.path = None
        self.num_rollouts = None
        self.index = None
        self.dataset = False
        self.data = None
        self.worker = None
        self.force_rewrite = force_rewrite
        self.run_name = None
        self.env_name = None
        self.ckpt = None
        self.seed = None
        self.policy_type = None
        self.env_creater = None
        self.require_activation = None
        self.is_es_agent = False
        self.agent = None

    def reset(
            self,
            ckpt,
            num_rollouts,
            seed,
            env_creater,
            run_name,
            env_name,
            require_activation=False,
    ):
        self.already_reset = False
        self.initialized = True
        self.path = get_dataset_path(ckpt, num_rollouts, seed)
        self.num_rollouts = num_rollouts
        self.require_activation = require_activation
        self.index = 0

        self.run_name = run_name
        self.ckpt = ckpt
        self.env_name = env_name
        self.env_creater = env_creater
        # self.policy = policy
        self.agent = None
        self.seed = seed

        # This is a workaround for ES agent.
        self.is_es_agent = False
        if self.run_name == "ES":
            self.is_es_agent = True

    def _lazy_reset(self):
        assert self.initialized
        if os.path.exists(self.path) and (not self.force_rewrite):
            print(
                "Dataset is detected! It's at {}."
                "\nWe will load data from it.".format(self.path)
            )
            try:
                with open(self.path, "rb") as f:
                    self.data = pickle.load(f)
                logging.critical(
                    "Dataset is detected!"
                    "\nWe have load data from <{}>.".format(self.path)
                )
            except Exception as e:
                logging.critical(
                    "Error detected when loading data from {}. "
                    "We will regenerate the broken pkl files.".format(
                        self.path
                    )
                )
                self.dataset = False
            else:
                self.dataset = True

        if (not os.path.exists(self.path)) or self.force_rewrite \
                or (not self.dataset):

            config_for_evaluation = {
                "batch_mode": "complete_episodes",
                "sample_batch_size": 1,
                "horizon": 3000,
                "seed": self.seed
            }

            if self.require_activation:
                self.agent = restore_agent_with_activation(
                    self.run_name, self.ckpt, self.env_name,
                    config_for_evaluation
                )
                self.policy_type = PPOTFPolicyWithActivation
                policy = self.agent.get_policy()
                assert "layer0" in policy.extra_compute_action_fetches(), \
                    "This policy is not we modified policy. Please use " \
                    "policy" \
                    "we reimplemented."
            else:
                if self.worker is not None:
                    try:
                        self.worker.stop()
                    except Exception:
                        pass
                self.agent = restore_agent(
                    self.run_name, self.ckpt, self.env_name,
                    config_for_evaluation
                )
                self.policy_type = PPOTFPolicy

            if hasattr(self.agent, "workers"):
                self.worker = self.agent.workers.local_worker()
                self.worker.set_weights(
                    {DEFAULT_POLICY_ID: self.agent.get_policy().get_weights()}
                )
            else:
                self.worker = None
                assert self.is_es_agent

            self.dataset = False
            self.data = []
            print("Successfully set weights for agent.")
        self.already_reset = True

    def wrap_sample(self):
        assert self.initialized
        if not self.already_reset:
            self._lazy_reset()
        result_list = []

        if self.is_es_agent:
            env = self.env_creater()

        for roll in range(self.num_rollouts):
            assert self.index < self.num_rollouts
            if self.dataset:
                result = self.data[self.index]
            else:
                if self.is_es_agent:
                    result = rollout(
                        self.agent,
                        env,
                        self.env_name,
                        require_trajectory=True,
                        require_extra_info=False
                    )
                    result = result['trajectory']
                else:
                    result = self.worker.sample()
                self.data.append(result)
            self.index += 1
            result_list.append(result)
        return result_list

    def get_dataset(self):
        assert self.initialized
        return self.dataset

    def close(self):
        assert self.initialized
        if self.dataset:
            logging.critical(
                "Data is already saved at: {} with len {}.".format(
                    self.path, len(self.data)
                )
            )
        else:
            with open(self.path, "wb") as f:
                pickle.dump(self.data, f)
            logging.critical(
                "Data is newly saved at: {} with len {}.".format(
                    self.path, len(self.data)
                )
            )
            self.dataset = len(self.data) >= self.num_rollouts
        # if self.worker:
        #     self.worker.stop()
        # if self.agent:
        #     self.agent.stop()
        return self.path


def make_worker(env_maker, ckpt, num_rollouts, seed, run_name, env_name):
    # assert agent._name == "PPO", "We only support ppo agent now!"
    # path = get_dataset_path(ckpt, num_rollout, seed)
    # if os.path.exists(path):
    #     return DataLoader(ckpt, num_rollout, seed)
    # policy = PPOTFPolicy
    worker = RolloutWorkerWrapper()
    worker.reset(
        ckpt=ckpt,
        num_rollouts=num_rollouts,
        seed=seed,
        env_creater=env_maker,
        run_name=run_name,
        env_name=env_name
    )
    return worker


def efficient_rollout_from_worker(worker, num_rollouts=None):
    trajctory_list = []
    t = time.time()
    data = worker.wrap_sample()
    print("[efficient_rollout_from_worker] Finish wrap_sample.")
    # data = copy.deepcopy(ray.get(obj_id))
    # data is a list. Each entry is a SampleBatch
    for sample_batch in data:
        if isinstance(sample_batch, list):
            trajectory = parse_es_rollout(sample_batch)
        else:
            trajectory = parse_single_rollout(sample_batch.data)
        trajctory_list.append(trajectory)
    logging.info(
        "Finish {} Rollouts. Cost: {} s.".format(
            num_rollouts or "--",
            time.time() - t
        )
    )
    return trajctory_list


def parse_rllib_trajectory_list(trajectory_list):
    return_list = []
    for num, trajectory in enumerate(trajectory_list):
        parsed_trajectory = parse_single_rollout(trajectory)
        return_list.append(parsed_trajectory)
    return trajectory_list


def parse_single_rollout(data):
    obs = data['obs']
    act = data['actions']
    rew = data['rewards']
    next_obs = data['new_obs']
    # value_function = data['vf_preds']
    done = data['dones']
    trajectory = [obs, act, next_obs, rew, done]
    return trajectory


def parse_es_rollout(data):
    assert len(data[0]) == 5
    obs = np.stack([t[0] for t in data])
    act = np.stack([t[1] for t in data])
    next_obs = np.stack([t[2] for t in data])
    reward = np.stack([t[3] for t in data])
    done = np.stack([t[4] for t in data])
    trajectory = [obs, act, next_obs, reward, done]
    return trajectory


"""
Modification:
    1. use iteration as the termination criterion
    2. pass an environment object which can be different from env_name
"""


def rollout(
        agent,
        env,
        env_name,
        num_steps=None,
        require_frame=False,
        require_trajectory=False,
        require_extra_info=False,
        require_full_frame=False,
        require_env_state=False,
        render_mode="rgb_array"
):
    assert require_frame or require_trajectory or require_extra_info or \
           require_env_state, "You must ask for some output!"

    if num_steps is None:
        num_steps = 3000

    policy_agent_mapping = default_policy_agent_mapping

    if hasattr(agent, "workers"):
        # env = agent.workers.local_worker().env
        multiagent = isinstance(env, MultiAgentEnv)
        if agent.workers.local_worker().multiagent:
            policy_agent_mapping = agent.config["multiagent"
                                                ]["policy_mapping_fn"]

        policy_map = agent.workers.local_worker().policy_map
        state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
        use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
        action_init = {
            p: m.action_space.sample()
            for p, m in policy_map.items()
        }
    else:
        # env = gym.make(env_name)
        multiagent = False
        policy = agent.policy
        state_init = {DEFAULT_POLICY_ID: None}
        use_lstm = {p: None for p, s in state_init.items()}
        action_init = {DEFAULT_POLICY_ID: policy.action_space.sample()}

    steps = 0
    now = time.time()
    start = now
    # while steps < (num_steps or steps + 1):
    for i in range(1):  # TODO for future extend to multiple iteration
        if require_trajectory:
            trajectory = []
        if require_frame:
            frames = []
            # assert env_name in ["BipedalWalker-v2"]
            frame_extra_info = {
                "value_function": [],
                "reward": [],
                "done": [],
                "step": [],
                "period_info": []
            }
        if require_extra_info:
            extra_infos = []
        if require_env_state:
            env_states = []

        mapping_cache = {}  # in case policy_agent_mapping is stochastic
        obs = env.reset()

        if require_env_state:
            env_states.append(copy.deepcopy(env.get_state_wrap()))

        agent_states = DefaultMapping(
            lambda agent_id: state_init[mapping_cache[agent_id]]
        )
        prev_actions = DefaultMapping(
            lambda agent_id: action_init[mapping_cache[agent_id]]
        )
        prev_rewards = collections.defaultdict(lambda: 0.)
        done = False
        reward_total = 0.0
        while not done and steps < (num_steps or steps + 1):
            if steps % LOG_INTERVAL_STEPS == (LOG_INTERVAL_STEPS - 1):
                logging.info(
                    "Current Steps: {}, Time Elapsed: {:.2f}s, "
                    "Last {} Steps Time: {:.2f}s".format(
                        steps,
                        time.time() - start, LOG_INTERVAL_STEPS,
                        time.time() - now
                    )
                )
                now = time.time()
            multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}
            action_dict = {}
            value_functions = {}
            for agent_id, a_obs in multi_obs.items():
                if a_obs is not None:
                    policy_id = mapping_cache.setdefault(
                        agent_id, policy_agent_mapping(agent_id)
                    )
                    p_use_lstm = use_lstm[policy_id]
                    if p_use_lstm:
                        a_action, p_state, a_info = agent.compute_action(
                            a_obs,
                            state=agent_states[agent_id],
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id
                        )
                        agent_states[agent_id] = p_state
                    else:
                        # This is a workaround
                        if agent._name == "ES":
                            a_action = agent.compute_action(a_obs)
                        else:
                            a_action, _, a_info = agent.compute_action(
                                a_obs,
                                prev_action=prev_actions[agent_id],
                                prev_reward=prev_rewards[agent_id],
                                policy_id=policy_id,
                                full_fetch=True
                            )
                    a_action = _flatten_action(a_action)  # tuple actions
                    action_dict[agent_id] = a_action
                    prev_actions[agent_id] = a_action
                    if require_extra_info:
                        extra_infos.append(a_info)
                    # This is a work around
                    if agent._name != "ES":
                        value_functions[agent_id] = a_info["vf_preds"]
            # This is a work around
            if require_frame and agent._name != "ES":
                frame_extra_info['value_function'].append(
                    value_functions[_DUMMY_AGENT_ID]
                )
            action = action_dict[_DUMMY_AGENT_ID]

            next_obs, reward, done, _ = env.step(action)

            if multiagent:
                done = done["__all__"]
                reward_total += sum(reward.values())
            else:
                reward_total += reward

            if require_frame:
                frame_extra_info["done"].append(done)
                frame_extra_info["reward"].append(reward_total)
                frame_extra_info["step"].append(steps)

                # data required for calculating period.
                # we observe the channel 7 and 12 which represent the speed
                # of the knee joints.
                # This only hold for BipedalWalker-v2.
                if env_name in ENV_NAME_PERIOD_FEATURE_LOOKUP.keys():
                    assert obs.ndim == 1
                    period_feature = ENV_NAME_PERIOD_FEATURE_LOOKUP[env_name]
                    frame_extra_info["period_info"].append(obs[period_feature])

            if multiagent:
                for agent_id, r in reward.items():
                    prev_rewards[agent_id] = r
            else:
                prev_rewards[_DUMMY_AGENT_ID] = reward

            kwargs = {"mode": render_mode if require_full_frame else "cropped"}
            # This copy() is really important!
            # Otherwise see error: pyarrow.lib.ArrowInvalid
            if require_frame:
                frame = env.render(**kwargs).copy()
                frames.append(frame)
            if require_trajectory:
                trajectory.append([obs, action, next_obs, reward, done])
            if require_env_state:
                env_states.append(copy.deepcopy(env.get_state_wrap()))
            steps += 1
            obs = next_obs
        logging.info("Episode reward", reward_total)
    result = {}
    if require_frame:
        result['frames'] = np.stack(frames)
        result['frame_extra_info'] = frame_extra_info
    if require_trajectory:
        result['trajectory'] = trajectory
    if require_extra_info:
        extra_info_dict = {k: [] for k in extra_infos[0].keys()}
        for item in extra_infos:
            for k, v in item.items():
                extra_info_dict[k].append(v)
        result["extra_info"] = extra_info_dict
    if require_env_state:
        result['env_states'] = env_states
    return result


def several_agent_rollout(
        yaml_path,
        num_rollouts,
        seed=0,
        num_workers=10,
        force_rewrite=False,
        return_data=False,
        require_activation=True,
        _num_agents=None
):
    name_ckpt_mapping = read_yaml(yaml_path, number=_num_agents)
    now_t_get = now_t = start_t = time.time()
    num_agents = len(name_ckpt_mapping)
    num_iteration = int(ceil(num_agents / num_workers))
    agent_ckpt_dict_range = list(name_ckpt_mapping.items())
    agent_count = 1
    agent_count_get = 1

    have_gpu = has_gpu()
    workers = [
        RolloutWorkerWrapper.as_remote(num_gpus=0.2 if have_gpu else 0
                                       ).remote(force_rewrite)
        for _ in range(num_workers)
    ]

    return_dict = {}

    for iteration in range(num_iteration):
        start = iteration * num_workers
        end = min((iteration + 1) * num_workers, num_agents)
        # obj_ids = []
        # workers = []
        obj_ids_dict = {}
        for i, (name, ckpt_dict) in \
                enumerate(agent_ckpt_dict_range[start:end]):
            ckpt = ckpt_dict["path"]
            env_name = ckpt_dict["env_name"]
            env_maker = ENV_MAKER_LOOKUP[env_name]
            run_name = ckpt_dict["run_name"]
            assert run_name == "PPO"

            # TODO Only support PPO now.
            workers[i].reset.remote(
                ckpt=ckpt,
                num_rollouts=num_rollouts,
                seed=seed,
                env_creater=env_maker,
                run_name=run_name,
                env_name=env_name,
                require_activation=require_activation
            )
            obj_id = workers[i].wrap_sample.remote()
            obj_ids_dict[name] = obj_id
            print(
                "[{}/{}] (+{:.1f}s/{:.1f}s) Start collect {} rollouts from "
                "agent"
                " <{}>".format(
                    agent_count, num_agents,
                    time.time() - now_t,
                    time.time() - start_t, num_rollouts, name
                )
            )

            agent_count += 1
            now_t = time.time()

        for (name, obj_id), worker in zip(obj_ids_dict.items(), workers):
            trajectory_list = copy.deepcopy(ray.get(obj_id))
            # for obj_id in obj_ids:
            #     trajectory_list.append(ray.get(obj_id))
            return_dict[name] = trajectory_list
            # worker.close.remote()
            print(
                "[{}/{}] (+{:.1f}s/{:.1f}s) Collected {} rollouts from agent"
                " <{}>".format(
                    agent_count_get, num_agents,
                    time.time() - now_t_get,
                    time.time() - start_t, num_rollouts, name
                )
            )
            agent_count_get += 1
            now_t_get = time.time()
    return return_dict if return_data else None


if __name__ == "__main__":
    # test_serveral_agent_rollout(True)
    # exit(0)
    # _test_es_agent_compatibility()
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml-path", required=True, type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-rollouts", '-n', type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--num-gpus", '-g', type=int, default=4)
    parser.add_argument("--force-rewrite", action="store_true")
    args = parser.parse_args()
    assert args.yaml_path.endswith("yaml")

    initialize_ray(num_gpus=args.num_gpus)
    several_agent_rollout(
        args.yaml_path, args.num_rollouts, args.seed, args.num_workers,
        args.force_rewrite
    )
