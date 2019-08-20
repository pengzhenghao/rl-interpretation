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

import gym
import numpy as np
import ray
import yaml
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.evaluation.episode import _flatten_action
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.tune.util import merge_dicts

from utils import DefaultMapping, VideoRecorder

VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080

from gym.envs.box2d.bipedal_walker import (
    BipedalWalker, VIEWPORT_H, VIEWPORT_W, SCALE, TERRAIN_HEIGHT, TERRAIN_STEP
)
from Box2D.b2 import circleShape

from opencv_wrappers import Surface
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


def scale_color(color_in_1):
    return tuple(int(c * 255) for c in color_in_1)


class OpencvViewer(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.surface = Surface(height=height, width=width)
        self.translation = 0, 0
        self.scale = 1, 1
        self.frame = np.empty((height, width, 4), dtype=np.uint8)
        self.frame.fill(255)

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scalex = self.width / (right - left)
        scaley = self.height / (top - bottom)
        self.translation = -left, -bottom
        self.scale = scalex, scaley

    def draw_circle(self, radius=10, res=30, filled=True, **attrs):
        raise NotImplementedError

    def translate(self, point):
        point1 = point[0] + self.translation[0], point[1] + \
                 self.translation[1]
        point2 = point1[0] * self.scale[0], point1[1] * \
                 self.scale[1]
        return self.height - point2[1], point2[0]

    def draw_polygon(self, v, filled=True, **attrs):
        v = [self.translate(p) for p in v]
        color = scale_color(attrs["color"])
        self.surface.polygon(v, color)

    def draw_polyline(self, v, **attrs):
        color = scale_color(attrs["color"])
        thickness = attrs['thickness'] if 'thickness' in attrs \
            else attrs['linewidth']
        for point1, point2 in zip(v[:-1], v[1:]):
            point1 = self.translate(tuple(point1))
            point2 = self.translate(tuple(point2))
            self.surface.line(point1, point2, color, thickness)

    def draw_line(self, start, end, **attrs):
        start = self.translate(start)
        end = self.translate(end)
        self.surface.line(start, end, **attrs)

    def render(self, return_rgb_array):
        self.frame.fill(255)
        # if not return_rgb_array:
        #     self.surface.display(1)
        frame = self.surface.raw_data()
        return frame[:, :, 2::-1]

    def close(self):
        del self.surface


class BipedalWalkerWrapper(BipedalWalker):
    def render(self, mode='rgb_array'):
        # This function is almost identical to the original one but the
        # importing of pyglet is avoided.
        if self.viewer is None:
            self.viewer = OpencvViewer(VIEWPORT_W, VIEWPORT_H)
        self.viewer.set_bounds(
            self.scroll, VIEWPORT_W / SCALE + self.scroll, 0,
                         VIEWPORT_H / SCALE
        )

        self.viewer.draw_polygon(
            [
                (self.scroll, 0),
                (self.scroll + VIEWPORT_W / SCALE, 0),
                (self.scroll + VIEWPORT_W / SCALE, VIEWPORT_H / SCALE),
                (self.scroll, VIEWPORT_H / SCALE),
            ],
            color=(0.9, 0.9, 1.0)
        )
        for poly, x1, x2 in self.cloud_poly:
            if x2 < self.scroll / 2: continue
            if x1 > self.scroll / 2 + VIEWPORT_W / SCALE: continue
            self.viewer.draw_polygon(
                [(p[0] + self.scroll / 2, p[1]) for p in poly],
                color=(1, 1, 1)
            )
        for poly, color in self.terrain_poly:
            if poly[1][0] < self.scroll: continue
            if poly[0][0] > self.scroll + VIEWPORT_W / SCALE: continue
            self.viewer.draw_polygon(poly, color=color)

        self.lidar_render = (self.lidar_render + 1) % 100
        i = self.lidar_render
        if i < 2 * len(self.lidar):
            l = self.lidar[i] if i < len(self.lidar
                                         ) else self.lidar[len(self.lidar) -
                                                           i - 1]
            self.viewer.draw_polyline(
                [l.p1, l.p2], color=(1, 0, 0), linewidth=1
            )

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    raise NotImplementedError
                    # t = rendering.Transform(translation=trans*f.shape.pos)
                    # self.viewer.draw_circle(f.shape.radius, 30,
                    # color=obj.color1).add_attr(t)
                    # self.viewer.draw_circle(f.shape.radius, 30,
                    # color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans * v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(
                        path, color=obj.color2, linewidth=2
                    )

        flagy1 = TERRAIN_HEIGHT
        flagy2 = flagy1 + 50 / SCALE
        x = TERRAIN_STEP * 3
        self.viewer.draw_polyline(
            [(x, flagy1), (x, flagy2)], color=(0, 0, 0), linewidth=2
        )
        f = [
            (x, flagy2), (x, flagy2 - 10 / SCALE),
            (x + 25 / SCALE, flagy2 - 5 / SCALE)
        ]
        self.viewer.draw_polygon(f, color=(0.9, 0.2, 0))
        self.viewer.draw_polyline(f + [f[0]], color=(0, 0, 0), linewidth=2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


class MultipolicyWrapper(MultiAgentEnv):
    def __init__(self, envs_dict, info_dict=None):
        self.envs_dict = envs_dict
        envs = list(envs_dict.values())
        self.metadata = envs[0].metadata
        self.action_space = [env.action_space for env in envs]
        self.observation_space = [env.observation_space for env in envs]
        self.num_envs = len(envs)
        self.info_dict = info_dict
        self.dones = {aid: False for aid in self.envs_dict.keys()}
        self.information = {}
        self._reset_information(self.info_dict)

    def _reset_information(self, info_dict=None):
        if info_dict:
            self.information = {}
            for info_name, info_details in info_dict.items():
                # info_datails should contain:
                #   default_value: the default value of this measurement
                #   text_function: a function transform the value to string
                #   pos_ratio: a tuple: (left_ratio, bottom_ratio)

                for key in ["default_value", "text_function", "pos_ratio"]:
                    assert key in info_details

                default_value = info_details['default_value']
                self.information[info_name] = \
                    {aid: default_value for aid in self.envs_dict.keys()}

                self.information[info_name].update(info_details)

            for key in self.envs_dict.keys():
                self.information["title"][key] = key

    def reset(self):
        self._reset_information(self.info_dict)
        self.dones = {aid: False for aid in self.envs_dict.keys()}
        ret = {}
        for aid, env in self.envs_dict.items():
            ret[aid] = env.reset()
        return ret

    def step(self, action_dict):
        ret_obs = {}
        ret_rew = {}
        ret_info = {}
        for aid, act in action_dict.items():
            if self.dones[aid]:
                continue
            obs, rew, done, info = self.envs_dict[aid].step(act)
            ret_obs[aid] = obs
            ret_rew[aid] = rew
            self.dones[aid] = done or self.dones[aid]
            self.information["done"][aid] = self.dones[aid]
            self.information["reward"][aid] += rew
            self.information["step"][aid] += 1
            ret_info[aid] = info
        ret_done = self.dones.copy()
        ret_done["__all__"] = all(ret_done.values())
        return ret_obs, ret_rew, ret_done, ret_info

    def update_value_functions(self, value_functions):
        if "value_function" in self.information:
            self.information["value_function"].update(value_functions)

    def render(self, *args, **kwargs):
        frames = {}
        for eid, env in self.envs_dict.items():
            frame = env.render(mode="rgb_array")
            frames[eid] = frame
        return frames


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
def collect_frames(run_name, env_name, config, ckpt, num_steps, seed=0):
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
    cls = get_agent_class(run_name)
    agent = cls(env=env_name, config=config)
    agent.restore(ckpt)
    # agents[name] = agent
    # print("[{}/{}] (T +{:.1f}s Total {:.1f}s) "
    #       "Restored agent <{}>".format(aid + 1, len(name_ckpt_mapping),
    #                                    time.time() - now,
    #                                    time.time() - start, name))
    # now = time.time()

    if hasattr(agent, "workers"):
        env = agent.workers.local_worker().env
    else:
        env = gym.make(env_name)

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
    extra_info = {"value": [], "reward": [], "done": [], "step": 0}

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

        extra_info['value'].append(value_functions[_DUMMY_AGENT_ID])
        action = action_dict[_DUMMY_AGENT_ID]
        obs, reward, done, _ = env.step(action)

        extra_info["done"].append(done)
        extra_info["reward"].append(reward)
        extra_info["step"] += 1

        prev_rewards[_DUMMY_AGENT_ID] = reward

        reward_total += reward

        kwargs = {"mode": "return_rgb_array"}
        frame = env.render(**kwargs)
        frames.append(frame)

    assert len(extra_info['done']) == len(extra_info["reward"]) == \
           len(extra_info["value"]) == len(frames)
    print("Episode reward", reward_total)
    return frames, extra_info


class GridVideoRecorder(object):
    # TODO make rollour async!

    def __init__(self, video_path, env_name, local_mode=False):

        ray.init(
            logging_level=logging.ERROR,
            log_to_driver=False,
            local_mode=local_mode
        )

        single_env = gym.make(env_name)

        self.video_recorder = VideoRecorder(video_path)

    def run(self, name_ckpt_mapping,
            num_steps=int(1e10),
            num_iters=1,
            seed=0,
            args_config=None):

        assert isinstance(name_ckpt_mapping, OrderedDict), \
            "The name-checkpoint dict is not OrderedDict!!! " \
            "We suggest you to use OrderedDict."

        # agents = OrderedDict()
        now = time.time()
        start = now
        for aid, (name, ckpt) in enumerate(name_ckpt_mapping.items()):
            ckpt = os.path.abspath(
                os.path.expanduser(ckpt))  # Remove relative dir
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

            frames, extra_info = \
                collect_frames.remote(run_name, env_name, config, ckpt,
                                      num_steps, seed)

            # if not env_name:
            #     if not config.get("env"):
            #         raise ValueError(
            #             "the following arguments are required: --env")
            #     env_name = config.get("env")
            #
            # cls = get_agent_class(run_name)
            # agent = cls(env=env_name, config=config)
            # agent.restore(ckpt)
            # agents[name] = agent
            print("[{}/{}] (T +{:.1f}s Total {:.1f}s) "
                  "Restored agent <{}>".format(aid + 1, len(name_ckpt_mapping),
                                               time.time() - now,
                                               time.time() - start, name))
            now = time.time()

            # if hasattr(agent, "workers"):
            #     env = agent.workers.local_worker().env
            # else:
            #     env = gym.make(env_name)
            #
            # env.seed(seed)

            # frames, extra_info = rollout(agent, env, num_steps)

    def _rollout(self, agent, num_steps):
        # Todo use env as input. Don't make env every time
        if hasattr(agent, "workers"):
            # env = agent.workers.local_worker().env
            policy_map = agent.workers.local_worker().policy_map
            state_init = {p: m.get_initial_state() for p, m in
                          policy_map.items()}
            use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
            action_init = {
                p: _flatten_action(m.action_space.sample())
                for p, m in policy_map.items()
            }
        else:
            # env = gym.make(env_name)
            use_lstm = {DEFAULT_POLICY_ID: False}

        frames = []
        extra_info = {"value": [], "reward": [], "done": [], "step": 0}

        mapping_cache = {}  # in case policy_agent_mapping is stochastic

        obs = self.env.reset()
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

            extra_info['value'].append(value_functions[_DUMMY_AGENT_ID])
            action = action_dict[_DUMMY_AGENT_ID]
            obs, reward, done, _ = self.env.step(action)

            extra_info["done"].append(done)
            extra_info["reward"].append(reward)
            extra_info["step"] += 1

            prev_rewards[_DUMMY_AGENT_ID] = reward

            reward_total += reward

            kwargs = {}
            frame = self.env.render(**kwargs)
            frames.append(frame)

        assert len(extra_info['done']) == len(extra_info["reward"]) == \
               len(extra_info["value"]) == len(frames)
        print("Episode reward", reward_total)
        return frames, extra_info

    def _aggregate(self):
        pass

    def close(self):

        ray.shutdown()


# def run(args, parser, name_ckpt_mapping):
def run(
        name_ckpt_mapping,
        video_path,
        run_name,
        num_steps=int(1e10),
        num_iters=1,
        seed=0,
        args_config=None,
        env_name=None,
        args_out=None,
        args_no_render=False,
        local_mode=False
):
    ray.init(
        logging_level=logging.ERROR,
        log_to_driver=False,
        local_mode=local_mode
    )

    assert isinstance(name_ckpt_mapping, OrderedDict), \
        "The name-checkpoint dict is not OrderedDict!!! " \
        "We suggest you to use OrderedDict."

    agents = OrderedDict()
    now = time.time()
    start = now
    for aid, (name, ckpt) in enumerate(name_ckpt_mapping.items()):
        ckpt = os.path.abspath(os.path.expanduser(ckpt))  # Remove relative dir
        config = {"log_level": "ERROR"}
        # Load configuration from file
        config_dir = os.path.dirname(ckpt)
        config_path = os.path.join(config_dir, "params.pkl")
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, "../params.pkl")
        if not os.path.exists(config_path):
            if not args_config:
                raise ValueError(
                    "Could not find params.pkl in either the checkpoint dir "
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
        if not env_name:
            if not config.get("env"):
                raise ValueError("the following arguments are required: --env")
            env_name = config.get("env")

        cls = get_agent_class(run_name)
        agent = cls(env=env_name, config=config)
        agent.restore(ckpt)
        agents[name] = agent
        print(
            "[{}/{}] (T +{:.1f}s Total {:.1f}s) Restored agent <{}>".format(
                aid + 1, len(name_ckpt_mapping),
                time.time() - now,
                time.time() - start, name
            )
        )
        now = time.time()

        if hasattr(agent, "workers"):
            env = agent.workers.local_worker().env
        else:
            env = gym.make(env_name)

        env.seed(seed)

        frames, extra_info = rollout(agent, env, num_steps)

    ray.shutdown()


def rollout(agent, env, num_steps=float("inf")):
    # Todo use env as input. Don't make env every time
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
    extra_info = {"value": [], "reward": [], "done": [], "step": 0}

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

        extra_info['value'].append(value_functions[_DUMMY_AGENT_ID])
        action = action_dict[_DUMMY_AGENT_ID]
        obs, reward, done, _ = env.step(action)

        extra_info["done"].append(done)
        extra_info["reward"].append(reward)
        extra_info["step"] += 1

        prev_rewards[_DUMMY_AGENT_ID] = reward

        reward_total += reward

        kwargs = {}
        frame = env.render(**kwargs)
        frames.append(frame)

    assert len(extra_info['done']) == len(extra_info["reward"]) == \
           len(extra_info["value"]) == len(frames)
    print("Episode reward", reward_total)
    return frames, extra_info


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    with open(args.yaml, 'r') as f:
        name_ckpt_list = yaml.safe_load(f)

    name_ckpt_mapping = OrderedDict()
    for d in name_ckpt_list:
        name_ckpt_mapping[d["name"]] = d["path"]

    # run(
    #     name_ckpt_mapping, args.yaml[:-5], args.run, args.steps, args.iters,
    #     args.seed, args.config, args.env, args.out, args.no_render,
    #     args.local_mode
    # )

    gvr = GridVideoRecorder(video_path=args.yaml[:-5],
                            env_name=args.env,
                            local_mode=args.local_mode)

    gvr.run(name_ckpt_mapping, args.steps, args.iters, args.seed)
