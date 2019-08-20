"""
Record video given a trained PPO model.

Usage:
    python record_video.py /YOUR_HOME/ray_results/EXP_NAME/TRAIL_NAME \
    -l 3000 --scene split -rf REWARD_FUNCTION_NAME
"""

from __future__ import absolute_import, division, print_function

import argparse
import collections
import json
import logging
import os
import pickle

import ray
import yaml
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.evaluation.episode import _flatten_action
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.tune.util import merge_dicts

from utils import DefaultMapping, VideoRecorder, BipedalWalkerWrapper

VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080

from collections import OrderedDict
import time

# info_datails should contain:
#   default_value: the default value of this measurement
#   text_function: a function transform the value to string
#   pos_ratio: a tuple: (left_ratio, bottom_ratio)
PRESET_INFORMATION_DICT = {
    "reward": {
        "default_value": 0.0,
        "text_function": lambda val: "Reward {:07.2f}".format(val),
        "pos_ratio": (0.95, 0.9)
    },
    "step": {
        "default_value": 0,
        "text_function": lambda val: "Step {}".format(val),
        "pos_ratio": (0.95, 0.8)
    },
    "done": {
        "default_value": False,
        "text_function": lambda val: "Done" if val else "",
        "pos_ratio": (0.25, 0.9)
    },
    "value_function": {
        "default_value": 0.0,
        "text_function": lambda val: "Value {:.3f}".format(val),
        "pos_ratio": (0.95, 0.7)
    },
    "title": {
        "default_value": "",
        "text_function": lambda val: val,
        "pos_ratio": (0.95, 0.05)
    }
}

ENVIRONMENT_MAPPING = {"BipedalWalker-v2": BipedalWalkerWrapper}


def create_parser(parser_creator=None):
    parser_creator = parser_creator or argparse.ArgumentParser
    parser = parser_creator(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Roll out a reinforcement learning agent "
        "given a checkpoint."
    )
    parser.add_argument(
        "yaml",
        type=str,
        help="yaml files contain name-checkpoint pairs from which to roll out."
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
        "--env", type=str, help="The gym environment to use.", required=True
    )
    parser.add_argument("--no-render", default=False, action="store_true")
    parser.add_argument("--num-envs", '-n', type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iters", default=1, type=int)
    parser.add_argument("--steps", default=int(1e10), type=int)
    parser.add_argument("--out", default=None, help="Output filename.")
    parser.add_argument("--local-mode", default=False, action="store_true")
    parser.add_argument(
        "--config",
        default="{}",
        type=json.loads,
        help="Algorithm-specific configuration (e.g. env, hyperparams). "
        "Surpresses loading of configuration from checkpoint."
    )
    return parser


@ray.remote
def collect_frames(
        run_name, env, env_name, config, ckpt, num_steps, num_iters=1, seed=0
):
    """
    This function create one agent and return one frame sequence.
    :param run_name:
    :param env_name:
    :param config:
    :param ckpt:
    :param num_steps:
    :param seed:
    :return:
    """
    # TODO allow multiple iters.

    cls = get_agent_class(run_name)
    agent = cls(env=env_name, config=config)
    agent.restore(ckpt)
    env.seed(seed)

    # frames, extra_info = rollout(agent, env, num_steps)
    if hasattr(agent, "workers"):
        # env = agent.workers.local_worker().env
        policy_map = agent.workers.local_worker().policy_map
        state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
        use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
        action_init = {
            p: _flatten_action(m.action_space.sample())
            for p, m in policy_map.items()
        }
    else:
        # env = gym.make(env_name)
        use_lstm = {DEFAULT_POLICY_ID: False}

    frames = []
    extra_info = {"value_function": [], "reward": [], "done": [], "step": 0}
    # extra_info = PRESET_INFORMATION_DICT

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
    cnt_steps = 0
    now = time.time()
    start = now
    while (not done) and (cnt_steps <= num_steps):
        if cnt_steps % 10 == 9:
            print(
                "Current Steps: {}, Time Elapsed: {:.2f}s, "
                "Last 10 Steps Time: {:.2f}s".format(
                    cnt_steps,
                    time.time() - start,
                    time.time() - now
                )
            )
            now = time.time()
        cnt_steps += 1
        multi_obs = {_DUMMY_AGENT_ID: obs}
        action_dict = {}
        value_functions = {}
        for agent_id, a_obs in multi_obs.items():
            if a_obs is not None:
                policy_id = mapping_cache.setdefault(
                    agent_id, DEFAULT_POLICY_ID
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
                value_functions[agent_id] = a_info["vf_preds"]

        extra_info['value_function'].append(value_functions[_DUMMY_AGENT_ID])
        action = action_dict[_DUMMY_AGENT_ID]
        obs, reward, done, _ = env.step(action)

        extra_info["done"].append(done)
        extra_info["reward"].append(reward)
        extra_info["step"] += 1

        prev_rewards[_DUMMY_AGENT_ID] = reward

        reward_total += reward

        kwargs = {"mode": "rgb_array"}
        # This copy() is really important!
        # Otherwise see error: pyarrow.lib.ArrowInvalid
        frame = env.render(**kwargs).copy()
        frames.append(frame)

    print("Episode reward", reward_total)
    env.close()
    return frames, extra_info


class GridVideoRecorder(object):
    def __init__(self, video_path, env_name, run_name, local_mode=False):

        ray.init(
            logging_level=logging.ERROR,
            log_to_driver=False,
            local_mode=local_mode
        )

        # single_env = gym.make(env_name)
        self.env_name = env_name
        self.run_name = run_name
        self.video_path = video_path
        # self.video_recorder = VideoRecorder(video_path)

    def rollout(
            self,
            name_ckpt_mapping,
            num_steps=int(1e10),
            num_iters=1,
            seed=0,
            args_config=None
    ):

        assert isinstance(name_ckpt_mapping, OrderedDict), \
            "The name-checkpoint dict is not OrderedDict!!! " \
            "We suggest you to use OrderedDict."

        # agents = OrderedDict()
        now = time.time()
        start = now
        object_id_dict = {}
        for aid, (name, ckpt) in enumerate(name_ckpt_mapping.items()):
            ckpt = os.path.abspath(
                os.path.expanduser(ckpt)
            )  # Remove relative dir
            config = {"log_level": "ERROR"}
            # Load configuration from file
            config_dir = os.path.dirname(ckpt)
            config_path = os.path.join(config_dir, "params.pkl")
            if not os.path.exists(config_path):
                config_path = os.path.join(config_dir, "../params.pkl")
            if not os.path.exists(config_path):
                if not args_config:
                    raise ValueError(
                        "Could not find params.pkl in either the checkpoint "
                        "dir "
                        "or "
                        "its parent directory."
                    )
            else:
                with open(config_path, "rb") as f:
                    config = pickle.load(f)
            if "num_workers" in config:
                config["num_workers"] = min(1, config["num_workers"])

            config["log_level"] = "ERROR"
            config = merge_dicts(config, args_config or {})

            object_id_dict[name] = collect_frames.remote(
                self.run_name,
                ENVIRONMENT_MAPPING[self.env_name],
                config,
                ckpt,
                num_steps,
                num_iters,
                seed
            )
            print(
                "[{}/{}] (T +{:.1f}s Total {:.1f}s) "
                "Restored agent <{}>".format(
                    aid + 1, len(name_ckpt_mapping),
                    time.time() - now,
                    time.time() - start, name
                )
            )
            now = time.time()

        frames_dict = {}
        extra_info_dict = PRESET_INFORMATION_DICT
        for aid, (name, object_id) in enumerate(object_id_dict.items()):
            frames, extra_info = ray.get(object_id)
            frames_dict[name] = frames
            for key, val in extra_info.items():
                extra_info_dict[key][name] = val
            extra_info_dict['title'][name] = name

            print(
                "[{}/{}] (T +{:.1f}s Total {:.1f}s) "
                "Get data from agent <{}>".format(
                    aid + 1, len(name_ckpt_mapping),
                    time.time() - now,
                    time.time() - start, name
                )
            )
            now = time.time()

        new_extra_info_dict = PRESET_INFORMATION_DICT
        for key in PRESET_INFORMATION_DICT.keys():
            new_extra_info_dict[key].update(extra_info_dict[key])

        return frames_dict, new_extra_info_dict

    def generate_video(self, frames_dict, extra_info_dict):
        self.video_recorder.generate_video(frames_dict, extra_info_dict)

    def close(self):
        ray.shutdown()


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    with open(args.yaml, 'r') as f:
        name_ckpt_list = yaml.safe_load(f)

    name_ckpt_mapping = OrderedDict()
    for d in name_ckpt_list:
        name_ckpt_mapping[d["name"]] = d["path"]

    gvr = GridVideoRecorder(
        video_path=args.yaml[:-5],
        env_name=args.env,
        run_name=args.run,
        local_mode=args.local_mode
    )

    frames_dict, extra_info_dict = gvr.rollout(
        name_ckpt_mapping, args.steps, args.iters, args.seed
    )

    gvr.close()
    vr = VideoRecorder("test_path_just_a_video", len(name_ckpt_mapping))
    vr.generate_video(frames_dict, extra_info_dict)
