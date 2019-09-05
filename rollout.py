"""
This filed is copied from ray.rllib.rollout but with little modification.
"""
# !/usr/bin/env python

from __future__ import absolute_import, division, print_function

import argparse
import collections
import json
import logging
import os
import pickle
import time
from math import ceil

import gym
import numpy as np
import ray
from ray.rllib.agents.ppo import PPOAgent
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.evaluation.episode import _flatten_action
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.tune.util import merge_dicts

from utils import restore_agent, initialize_ray, ENV_MAKER_LOOKUP, restore_agent_with_activation, has_gpu
from process_data import read_yaml
from tf_model import PPOTFPolicyWithActivation

EXAMPLE_USAGE = """
Example Usage via RLlib CLI:
    rllib rollout /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl

Example Usage via executable:
    ./rollout.py /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl
"""

# Note: if you use any custom models or envs, register them here first, e.g.:
#
# ModelCatalog.register_custom_model("pa_model", ParametricActionsModel)
# register_env("pa_cartpole", lambda _: ParametricActionCartpole(10))

LOG_INTERVAL_STEPS = 500


def create_parser(parser_creator=None):
    parser_creator = parser_creator or argparse.ArgumentParser
    parser = parser_creator(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Roll out a reinforcement learning agent "
        "given a checkpoint.",
        epilog=EXAMPLE_USAGE
    )

    parser.add_argument(
        "checkpoint", type=str, help="Checkpoint from which to roll out."
    )
    required_named = parser.add_argument_group("required named arguments")
    required_named.add_argument(
        "--run",
        type=str,
        required=True,
        help="The algorithm or model to train. This may refer to the name "
        "of a built-on algorithm (e.g. RLLib's DQN or PPO), or a "
        "user-defined trainable function or class registered in the "
        "tune registry."
    )
    required_named.add_argument(
        "--env", type=str, help="The gym environment to use."
    )
    parser.add_argument(
        "--no-render",
        default=False,
        action="store_const",
        const=True,
        help="Surpress rendering of the environment."
    )
    parser.add_argument(
        "--steps", default=10000, help="Number of steps to roll out."
    )
    parser.add_argument("--out", default=None, help="Output filename.")
    parser.add_argument(
        "--config",
        default="{}",
        type=json.loads,
        help="Algorithm-specific configuration (e.g. env, hyperparams). "
        "Surpresses loading of configuration from checkpoint."
    )
    return parser


def run(args, parser):
    config = {}
    # Load configuration from file
    config_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(config_dir, "params.pkl")
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")
    if not os.path.exists(config_path):
        if not args.config:
            raise ValueError(
                "Could not find params.pkl in either the checkpoint dir or "
                "its parent directory."
            )
    else:
        with open(config_path, "rb") as f:
            config = pickle.load(f)
    if "num_workers" in config:
        config["num_workers"] = min(2, config["num_workers"])
    config = merge_dicts(config, args.config)
    if not args.env:
        if not config.get("env"):
            parser.error("the following arguments are required: --env")
        args.env = config.get("env")

    ray.init()

    cls = get_agent_class(args.run)
    agent = cls(env=args.env, config=config)
    agent.restore(args.checkpoint)
    num_steps = int(args.steps)
    rollout(agent, args.env, num_steps, args.out, args.no_render)


class DefaultMapping(collections.defaultdict):
    """default_factory now takes as an argument the missing key."""

    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value


def default_policy_agent_mapping(unused_agent_id):
    return DEFAULT_POLICY_ID


# def test_efficient_rollout():
#     initialize_ray()
#     agent = PPOAgent(env="BipedalWalker-v2")
#     env = gym.make("BipedalWalker-v2")
#     w = make_worker(lambda _: env, 0)
#     set_weight(w, agent)
#     ret = efficient_rollout(w, 7)
#     print("start for another time!")
#     ret = efficient_rollout(w, 7)
#     return ret
# return efficient_rollout(agent, env, 3)


def get_dataset_path(ckpt, num_rollout, seed):
    ckpt = os.path.abspath(os.path.expanduser(ckpt))
    return "{}_rollout{}_seed{}.pkl".format(ckpt, num_rollout, seed)


def on_episode_start(info):
    # print(info.keys())  # -> "env", 'episode"
    episode = info["episode"]
    # print("episode {} started".format(episode.episode_id))
    episode.user_data["last_layer_activation"] = []


def on_episode_step(info):
    episode = info["episode"]
    pole_angle = abs(episode.last_observation_for()[2])
    episode.user_data["last_layer_activation"].append(pole_angle)


from tf_model import model_config, register


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

    def reset(
            self,
            ckpt,
            num_rollouts,
            seed,
            env_creater,
            # policy_type,
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
        # assert self.initialized
        # if self.dataset:
        #     return
        self.run_name = run_name
        self.ckpt = ckpt
        self.env_name = env_name
        self.env_creater = env_creater
        # self.policy = policy
        self.seed = seed

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
                logging.critical("Error detected when loading data from {}. "
                              "We will regenerate the broken pkl files."
                              .format(self.path))
                self.dataset = False
            else:
                self.dataset = True

        if (not os.path.exists(self.path)) or self.force_rewrite \
                or (not self.dataset):
            if self.require_activation:
                agent = restore_agent_with_activation(
                    self.run_name, self.ckpt, self.env_name
                )
                self.policy_type = PPOTFPolicyWithActivation
            else:
                agent = restore_agent(self.run_name, self.ckpt, self.env_name)
                self.policy_type = PPOTFPolicy
            policy = agent.get_policy()
            if self.require_activation:
                assert "layer0" in policy.extra_compute_action_fetches(), \
                "This policy is not we modified policy. Please use policy we " \
                "reimplemented."

            batch_mode = "complete_episodes"
            batch_steps = 1
            episode_horizon = 3000
            sample_async = True

            register()
            self.worker = RolloutWorker(
                env_creator=self.env_creater,
                policy=self.policy_type,
                batch_mode=batch_mode,
                batch_steps=batch_steps,
                episode_horizon=episode_horizon,
                sample_async=sample_async,
                seed=self.seed,
                model_config=model_config if self.require_activation else None,
                log_level="INFO"
            )
            self.dataset = False
            self.data = []

            self.worker.set_weights(
                {DEFAULT_POLICY_ID: agent.get_policy().get_weights()}
            )

            print("Successfully set weights for agent.")
        self.already_reset = True


    def wrap_sample(self, return_none=False):
        assert self.initialized

        if not self.already_reset:
            self._lazy_reset()

        assert self.index < self.num_rollouts
        if self.dataset:
            result = self.data[self.index]
        else:
            result = self.worker.sample()
            self.data.append(result)
        self.index += 1
        if return_none:
            return None
        return result

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
        return self.path


def make_worker(env_maker, ckpt, num_rollout, seed, run_name, env_name):
    # assert agent._name == "PPO", "We only support ppo agent now!"
    # path = get_dataset_path(ckpt, num_rollout, seed)
    # if os.path.exists(path):
    #     return DataLoader(ckpt, num_rollout, seed)
    policy = PPOTFPolicy
    worker = RolloutWorkerWrapper.as_remote().remote()
    worker.reset.remote(
        ckpt, num_rollout, seed, env_maker, policy, run_name, env_name
    )
    return worker


def efficient_rollout_from_worker(worker, num_rollouts):
    trajctory_list = []
    obj_ids = []
    t = time.time()
    for num in range(num_rollouts):
        obj_ids.append(worker.wrap_sample.remote())
    for num, obj in enumerate(obj_ids):
        # We have so many information:
        # dict_keys(['t', 'eps_id', 'agent_index', 'obs', 'actions',
        # 'rewards', 'prev_actions', 'prev_rewards', 'dones', 'infos',
        # 'new_obs', 'action_prob', 'vf_preds', 'behaviour_logits',
        # 'unroll_id', 'advantages', 'value_targets'])
        data = ray.get(obj).data
        trajectory = parse_single_rollout(data)
        logging.info(
            "Finish collect {}/{} rollouts. The latest rollout contain {}"
            " steps.".format(
                num + 1,
                num_rollouts,
                len(trajectory[0]),
            )
        )
        trajctory_list.append(trajectory)
    logging.info(
        "Finish {} Rollouts. Cost: {} s.".format(
            num_rollouts,
            time.time() - t
        )
    )
    worker.close.remote()
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


"""
Modification:
    1. use iteration as the termination criterion
    2. pass an environment object which can be different from env_name
"""


def rollout(
        agent,
        env,
        num_steps=None,
        require_frame=False,
        require_trajectory=False,
        require_extra_info=False,
        require_full_frame=False
):
    assert require_frame or require_trajectory or require_extra_info, \
        "You must ask for some output!"

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
            frame_extra_info = {
                "value_function": [],
                "reward": [],
                "done": [],
                "step": []
            }
        if require_extra_info:
            extra_infos = []

        mapping_cache = {}  # in case policy_agent_mapping is stochastic
        obs = env.reset()
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

            if multiagent:
                for agent_id, r in reward.items():
                    prev_rewards[agent_id] = r
            else:
                prev_rewards[_DUMMY_AGENT_ID] = reward

            kwargs = {"mode": "rgb_array" if require_full_frame else "cropped"}
            # This copy() is really important!
            # Otherwise see error: pyarrow.lib.ArrowInvalid
            if require_frame:
                frame = env.render(**kwargs).copy()
                frames.append(frame)
            if require_trajectory:
                trajectory.append([obs, action, next_obs, reward, done])
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
    return result


def replay(trajectory, agent):
    policy = agent.get_policy(DEFAULT_POLICY_ID)
    obs_batch = [tansition[0] for tansition in trajectory]
    obs_batch = np.asarray(obs_batch)
    actions, _, infos = policy.compute_actions(obs_batch)
    return actions, infos


def _test_es_agent_compatibility():
    from ray.rllib.agents.es import ESTrainer
    es = ESTrainer(env="BipedalWalker-v2")
    env = gym.make("BipedalWalker-v2")
    rollout(es, env, num_steps=100, require_frame=True)


def test_RolloutWorkerWrapper():
    initialize_ray(test_mode=True)
    env_maker = lambda _: gym.make("BipedalWalker-v2")
    ckpt = "test/fake-ckpt1/checkpoint-313"
    # rww = RolloutWorkerWrapper(ckpt, 2, 0, env_maker, PPOTFPolicy)
    # for _ in range(2):
    #     result = rww.wrap_sample()
    # print(result)
    # rww.close()

    rww_new = RolloutWorkerWrapper.as_remote().remote(
        ckpt,
        2,
        0,
        env_maker,
        PPOTFPolicy,
        run_name="PPO",
        env_name="BipedalWalker-v2"
    )
    for _ in range(2):
        result = ray.get(rww_new.wrap_sample.remote())
    print(result)
    print("Prepare to close")
    print("Dataset: ", ray.get(rww_new.get_dataset.remote()))
    rww_new.close.remote()
    print("After close")


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
                ckpt, num_rollouts, seed, env_maker, run_name, env_name,
                require_activation
            )
            obj_ids = []
            for _ in range(num_rollouts):
                # Ask to return None.
                obj_ids.append(workers[i].wrap_sample.remote(not return_data))
            obj_ids_dict[name] = obj_ids
            print(
                "[{}/{}] (+{:.1f}s/{:.1f}s) Start collect {} rollouts from agent"
                " <{}>".format(
                    agent_count, num_agents,
                    time.time() - now_t,
                    time.time() - start_t, num_rollouts, name
                )
            )

            agent_count += 1
            now_t = time.time()

        for (name, obj_ids), worker in zip(obj_ids_dict.items(), workers):
            trajectory_list = []
            for obj_id in obj_ids:
                trajectory_list.append(ray.get(obj_id))
            return_dict[name] = trajectory_list
            worker.close.remote()
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


def _replay(obs, run_name, ckpt, env_name):
    agent = restore_agent_with_activation(run_name, ckpt, env_name)
    act, _, infos = agent.get_policy().compute_actions(obs)
    return act, infos


@ray.remote(num_gpus=0.2)
def remote_replay_gpu(obs, run_name, ckpt, env_name):
    return _replay(obs, run_name, ckpt, env_name)


@ray.remote
def remote_replay_cpu(obs, run_name, ckpt, env_name):
    return _replay(obs, run_name, ckpt, env_name)


def several_agent_replay(
        yaml_path,
        obs,
        # num_rollouts,
        seed=0,
        num_workers=10,
        _num_agents=None
        # force_rewrite=False,
        # return_data=False
):
    name_ckpt_mapping = read_yaml(yaml_path, number=_num_agents)
    now_t_get = now_t = start_t = time.time()
    num_agents = len(name_ckpt_mapping)
    num_iteration = int(ceil(num_agents / num_workers))
    agent_ckpt_dict_range = list(name_ckpt_mapping.items())
    agent_count = 1
    agent_count_get = 1

    have_gpu = has_gpu()
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
                # if "env_name" in ckpt_dict else "BipedalWalker-v2"
            # env_maker = ENV_MAKER_LOOKUP[env_name]
            run_name = ckpt_dict["run_name"]
                # if "run_name" in ckpt_dict else "PPO"
            assert run_name == "PPO"

            if have_gpu:
                obj_id = remote_replay_gpu.remote(
                    obs, run_name, ckpt, env_name
                )
            else:
                obj_id = remote_replay_cpu.remote(
                    obs, run_name, ckpt, env_name
                )
            obj_ids_dict[name] = obj_id

            print(
                "[{}/{}] (+{:.1f}s/{:.1f}s) Start collect output of {} samples"
                " from agent <{}>".format(
                    agent_count, num_agents,
                    time.time() - now_t,
                    time.time() - start_t, obs.shape, name
                )
            )

            agent_count += 1
            now_t = time.time()

        for agent_name, obj_id in obj_ids_dict.items():
            act, infos = ray.get(obj_id)
            return_dict[agent_name] = {"act": act, "infos": infos}

            # trajectory_list = []
            # for obj_id in obj_ids:
            #     trajectory_list.append(ray.get(obj_id))
            # return_dict[name] = trajectory_list
            # worker.close.remote()
            print(
                "[{}/{}] (+{:.1f}s/{:.1f}s) Collected output of {} samples "
                "from agent <{}>".format(
                    agent_count_get, num_agents,
                    time.time() - now_t_get,
                    time.time() - start_t, obs.shape, agent_name
                )
            )
            agent_count_get += 1
            now_t_get = time.time()
    return return_dict


def test_RolloutWorkerWrapper_with_activation():
    initialize_ray(test_mode=True)
    env_maker = lambda _: gym.make("BipedalWalker-v2")
    ckpt = "test/fake-ckpt1/checkpoint-313"
    rww_new = RolloutWorkerWrapper.as_remote().remote(True)
    rww_new.reset.remote(
        ckpt, 2, 0, env_maker, "PPO", "BipedalWalker-v2", True
    )
    for _ in range(2):
        result = ray.get(rww_new.wrap_sample.remote())
    print(result)
    print("Prepare to close")
    print("Dataset: ", ray.get(rww_new.get_dataset.remote()))
    rww_new.close.remote()
    print("After close")
    return result


def test_serveral_agent_rollout(force=False):
    yaml_path = "data/0811-random-test.yaml"
    num_rollouts = 2
    initialize_ray()
    return several_agent_rollout(
        yaml_path, num_rollouts, force_rewrite=force, return_data=True
    )


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
