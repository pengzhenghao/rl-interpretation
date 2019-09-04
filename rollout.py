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

from utils import restore_agent, initialize_ray

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


def test_efficient_rollout():
    initialize_ray()
    agent = PPOAgent(env="BipedalWalker-v2")
    env = gym.make("BipedalWalker-v2")
    w = make_worker(lambda _: env, agent, 0)
    set_weight(w, agent)
    ret = efficient_rollout(w, 7)
    print("start for another time!")
    ret = efficient_rollout(w, 7)
    return ret
    # return efficient_rollout(agent, env, 3)


def get_dataset_path(ckpt, num_rollout, seed):
    ckpt = os.path.abspath(os.path.expanduser(ckpt))
    return "{}_rollout{}_seed{}.pkl".format(ckpt, num_rollout, seed)


class RolloutWorkerWrapper(RolloutWorker):
    @classmethod
    def as_remote(cls, num_cpus=None, num_gpus=None, resources=None):
        return ray.remote(
            num_cpus=num_cpus, num_gpus=num_gpus, resources=resources
        )(cls)

    def __init__(
            self,
            ckpt,
            num_rollouts,
            seed,
            env_creater,
            policy,
            batch_mode="complete_episodes",
            batch_steps=1,
            episode_horizon=3000,
            sample_async=True
    ):
        self.path = get_dataset_path(ckpt, num_rollouts, seed)
        self.num_rollouts = num_rollouts
        self.index = 0
        if os.path.exists(self.path):
            print(
                "Dataset is detected! It's at {}."
                "\nWe will load data from it.".format(self.path)
            )
            with open(self.path, "rb") as f:
                self.data = pickle.load(f)
                self.dataset = True
        else:
            super(RolloutWorkerWrapper, self).__init__(
                env_creator=env_creater,
                policy=policy,
                batch_mode=batch_mode,
                batch_steps=batch_steps,
                episode_horizon=episode_horizon,
                sample_async=sample_async,
                seed=seed
            )
            self.dataset = False
            self.data = []

    def wrap_sample(self):
        assert self.index < self.num_rollouts
        if self.dataset:
            result = self.data[self.index]
        else:
            result = self.sample()
            self.data.append(result)
        self.index += 1
        return result

    def get_dataset(self):
        return self.dataset

    def close(self):
        if self.dataset:
            print(
                "Data is already saved at: {} with len {}.".format(
                    self.path, len(self.data)
                )
            )
        else:
            with open(self.path, "wb") as f:
                pickle.dump(self.data, f)
            print(
                "Data is newly saved at: {} with len {}.".format(
                    self.path, len(self.data)
                )
            )
            self.dataset = len(self.data) >= self.num_rollouts


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
        ckpt, 2, 0, env_maker, PPOTFPolicy
    )
    for _ in range(2):
        result = ray.get(rww_new.wrap_sample.remote())
    print(result)
    print("Prepare to close")
    print("Dataset: ", ray.get(rww_new.get_dataset.remote()))
    rww_new.close.remote()
    print("After close")


def make_worker(env_maker, agent, ckpt, num_rollout, seed):
    assert agent._name == "PPO", "We only support ppo agent now!"
    # path = get_dataset_path(ckpt, num_rollout, seed)
    # if os.path.exists(path):
    #     return DataLoader(ckpt, num_rollout, seed)
    policy = PPOTFPolicy
    worker = RolloutWorkerWrapper.as_remote().remote(
        ckpt, num_rollout, seed, env_maker, policy
    )
    return worker


def set_weight(worker, agent):
    if ray.get(worker.get_dataset.remote()):
        return
    # assert agent._name == "PPO", "We only support ppo agent now!"
    # policy = PPOTFPolicy
    worker.set_weights.remote(
        {DEFAULT_POLICY_ID: agent.get_policy().get_weights()}
    )


def efficient_rollout(worker, num_rollouts):
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
        obs = data['obs']
        act = data['actions']
        rew = data['rewards']
        next_obs = data['new_obs']
        # value_function = data['vf_preds']
        done = data['dones']
        logging.info(
            "Finish collect {}/{} rollouts. The latest rollout contain {}"
            " steps.".format(
                num + 1,
                num_rollouts,
                len(obs),
            )
        )
        trajectory = [obs, act, next_obs, rew, done]
        trajctory_list.append(trajectory)
    logging.info(
        "Finish {} Rollouts. Cost: {} s.".format(
            num_rollouts,
            time.time() - t
        )
    )
    worker.close.remote()
    return trajctory_list


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


if __name__ == "__main__":
    # This part is used for test only!
    # Don't call python rollout.py for any other purpose.
    test_RolloutWorkerWrapper()
    # initialize_ray(False)
    # env = gym.make("CartPole-v0")
    # ret = rollout(
    #     restore_agent("PPO", None, "CartPole-v0"),
    #     env,
    #     100,
    #     require_extra_info=True,
    #     require_trajectory=True,
    #     require_frame=True
    # )
    # print(ret)
    #
    # _test_es_agent_compatibility()
