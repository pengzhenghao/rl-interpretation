"""
Record video given a trained PPO model.

Usage:
    python record_video.py /YOUR_HOME/ray_results/EXP_NAME/TRAIL_NAME \
    -l 3000 --scene split -rf REWARD_FUNCTION_NAME
"""

from __future__ import absolute_import, division, print_function, \
    absolute_import, division, print_function

import argparse
import json
import logging
from math import ceil

import ray
import yaml

from utils import build_config, VideoRecorder, BipedalWalkerWrapper, \
    restore_agent

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
# from gym.envs.box2d import BipedalWalker
# ENVIRONMENT_MAPPING = {"BipedalWalker-v2": BipedalWalker}
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


from rollout import rollout


@ray.remote
def collect_frames(
        run_name,
        env_maker,
        env_name,
        config,
        ckpt,
        num_steps=0,
        num_iters=1,
        seed=0
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

    agent = restore_agent(run_name, ckpt, env_name, config)
    env = env_maker()
    env.seed(seed)

    result = rollout(agent, env, num_steps, require_frame=True)
    frames, extra_info = result['frames'], result['extra_info']
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

    def generate_frames(
            self,
            name_ckpt_mapping,
            num_steps=int(1e10),
            num_iters=1,
            seed=0,
            args_config=None,
            num_workers=10
    ):

        assert isinstance(name_ckpt_mapping, OrderedDict), \
            "The name-checkpoint dict is not OrderedDict!!! " \
            "We suggest you to use OrderedDict."

        num_agent = len(name_ckpt_mapping)

        num_iteration = int(ceil(num_agent / num_workers))

        name_ckpt_mapping_range = list(name_ckpt_mapping.items())
        agent_count = 1

        frames_dict = {}
        extra_info_dict = PRESET_INFORMATION_DICT

        for iteration in range(num_iteration):
            print("We should stop here and wait!")
            idx_start = iteration * num_workers
            idx_end = min((iteration + 1) * num_workers, num_agent)

            now = time.time()
            start = now
            object_id_dict = {}
            for incre, (name, ckpt) in \
                    enumerate(name_ckpt_mapping_range[idx_start: idx_end]):
                config = build_config(ckpt, args_config)
                object_id_dict[name] = collect_frames.remote(
                    self.run_name, ENVIRONMENT_MAPPING[self.env_name],
                    self.env_name, config, ckpt, num_steps, num_iters, seed
                )
                print(
                    "[{}/{}] (T +{:.1f}s Total {:.1f}s) "
                    "Restored agent <{}>".format(
                        agent_count + incre, len(name_ckpt_mapping),
                        time.time() - now,
                        time.time() - start, name
                    )
                )
                now = time.time()

            for incre, (name, object_id) in enumerate(object_id_dict.items()):
                frames, extra_info = ray.get(object_id)
                frames_dict[name] = frames
                for key, val in extra_info.items():
                    extra_info_dict[key][name] = val
                extra_info_dict['title'][name] = name

                print(
                    "[{}/{}] (T +{:.1f}s Total {:.1f}s) "
                    "Get data from agent <{}>".format(
                        incre + agent_count, len(name_ckpt_mapping),
                        time.time() - now,
                        time.time() - start, name
                    )
                )
                now = time.time()

            agent_count += num_workers

        new_extra_info_dict = PRESET_INFORMATION_DICT
        for key in PRESET_INFORMATION_DICT.keys():
            new_extra_info_dict[key].update(extra_info_dict[key])
        return frames_dict, new_extra_info_dict

    def generate_video(self, frames_dict, extra_info_dict):
        vr = VideoRecorder(self.video_path, len(frames_dict))
        vr.generate_video(frames_dict, extra_info_dict)

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

    frames_dict, extra_info_dict = gvr.generate_frames(
        name_ckpt_mapping, args.steps, args.iters, args.seed
    )

    gvr.generate_video(frames_dict, extra_info_dict)

    gvr.close()
