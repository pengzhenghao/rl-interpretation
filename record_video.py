"""
Record video given a trained PPO model.

Usage:
    python record_video.py /YOUR_HOME/ray_results/EXP_NAME/TRAIL_NAME \
    -l 3000 --scene split -rf REWARD_FUNCTION_NAME
"""

from __future__ import absolute_import, division, print_function, \
    absolute_import, division, print_function

import argparse
import copy
import json
from math import ceil

import ray
import yaml

from process_data import get_name_ckpt_mapping
from rollout import rollout
from utils import build_config, VideoRecorder, BipedalWalkerWrapper, \
    restore_agent, initialize_ray

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


@ray.remote
class CollectFramesWorker(object):
    def __init__(
            self, run_name, env_maker, env_name, num_steps, num_iters, seed
    ):
        self.run_name = run_name
        self.env_maker = env_maker
        self.env_name = env_name
        self.num_steps = num_steps
        self.num_iters = num_iters
        self.seed = seed

    def collect_frames(self, config, ckpt):
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

        agent = restore_agent(self.run_name, ckpt, self.env_name, config)
        env = self.env_maker()
        env.seed(self.seed)

        result = rollout(agent, env, self.num_steps, require_frame=True)
        frames, extra_info = result['frames'], result['frame_extra_info']
        env.close()
        return frames, extra_info


class GridVideoRecorder(object):
    def __init__(self, video_path, env_name, run_name, local_mode=False):
        initialize_ray(local_mode)

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
            name_column_mapping=None,
            name_row_mapping=None,
            name_loc_mapping=None,
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

        workers = [
            CollectFramesWorker.remote(
                self.run_name, ENVIRONMENT_MAPPING[self.env_name],
                self.env_name, num_steps, num_iters, seed
            ) for _ in range(num_workers)
        ]

        for iteration in range(num_iteration):
            idx_start = iteration * num_workers
            idx_end = min((iteration + 1) * num_workers, num_agent)

            now = time.time()
            start = now
            object_id_dict = {}
            for incre, (name, ckpt) in \
                    enumerate(name_ckpt_mapping_range[idx_start: idx_end]):
                config = build_config(ckpt, args_config)
                object_id_dict[name] = workers[incre].collect_frames.remote(
                    config, ckpt
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

                # To avoid memory leakage. This part is really important!
                new_frames = copy.deepcopy(frames)
                new_extra_info = copy.deepcopy(extra_info)
                # del frames
                # del extra_info

                frames_info = {
                    "frames":
                    new_frames,
                    "column":
                    None if name_column_mapping is None else
                    name_column_mapping[name],
                    "row":
                    None
                    if name_row_mapping is None else name_row_mapping[name],
                    "loc":
                    None
                    if name_loc_mapping is None else name_loc_mapping[name]
                }

                frames_dict[name] = frames_info
                for key, val in new_extra_info.items():
                    if key in extra_info_dict:
                        extra_info_dict[key][name] = val
                    elif key == "vf_preds":
                        extra_info_dict["value_function"][name] = val
                extra_info_dict['title'][name] = name

                print(
                    "[{}/{}] (T +{:.1f}s Total {:.1f}s) "
                    "Got data from agent <{}>".format(
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
        new_extra_info_dict['frame_info'] = {
            "width": new_frames[0].shape[1],
            "height": new_frames[0].shape[0]
        }
        return frames_dict, new_extra_info_dict

    def generate_video(self, frames_dict, extra_info_dict):
        print(
            "Start generating grid containing {} videos.".format(
                len(frames_dict)
            )
        )
        locations = [f_info['loc'] for f_info in frames_dict.values()]
        assert len(set(locations)) == len(locations)

        max_row = max([row + 1 for row, _ in locations])
        max_col = max([col + 1 for _, col in locations])

        vr = VideoRecorder(
            self.video_path, grids={
                'col': max_col,
                'row': max_row
            }
        )
        vr.generate_video(frames_dict, extra_info_dict)

    def close(self):
        ray.shutdown()


def _build_name_row_mapping(cluster_dict):
    """
    cluster_dict = {name: {"distance": float, "cluster": int}, ..}
    :param cluster_dict:
    :return:
    """
    ret = {}
    for name, cluster_info in cluster_dict.items():
        ret[name] = "Cluster {}".format(cluster_info['cluster'])
    return ret


def _build_name_col_mapping(cluster_dict):
    # We haven't found the suitable names for each column.
    return None


def _transform_name_ckpt_mapping(
        name_ckpt_mapping, prediction, max_num_cols=10
):
    # Function:
    # 1. re-order the OrderedDict
    # 2. add distance information in name
    clusters = set([c_info['cluster'] for c_info in prediction.values()])

    new_name_ckpt_mapping = OrderedDict()

    old_row_mapping = _build_name_row_mapping(prediction)
    old_col_mapping = _build_name_col_mapping(prediction)

    name_loc_mapping = {}
    name_row_mapping = {} if old_row_mapping is not None else None
    name_col_mapping = {} if old_col_mapping is not None else None

    for row_id, cls in enumerate(clusters):
        within_one_cluster = {
            k: v
            for k, v in prediction.items() if v['cluster'] == cls
        }
        pairs = sorted(
            within_one_cluster.items(), key=lambda kv: kv[1]['distance']
        )
        if len(pairs) > max_num_cols:
            pairs = pairs[:max_num_cols]
        # {'cc': {'distance': 1}, 'aa': {'distance': 10}, ..}
        row_cluster_dict = dict(pairs)
        for col_id, (name, cls_info) in enumerate(row_cluster_dict.items()):
            loc = (row_id, col_id)

            components = name.split(" ")
            new_name = components[0]
            for com in components:
                if "=" not in com:
                    continue
                # We expect com to be like "seed=10" or "rew=201"
                # Then we transform it to "s10" or "r201"
                new_name += "," + com.split('=')[0][0] + com.split('=')[1]

            new_name = new_name + " d{:.2f}".format(cls_info['distance'])
            new_name_ckpt_mapping[new_name] = name_ckpt_mapping[name]
            name_loc_mapping[new_name] = loc
            if old_row_mapping is not None:
                name_row_mapping[new_name] = old_row_mapping[name]
            if old_col_mapping is not None:
                name_col_mapping[new_name] = old_col_mapping[name]

    return new_name_ckpt_mapping, name_loc_mapping, name_row_mapping, \
           name_col_mapping


def generate_video_of_cluster(
        prediction,
        env_name,
        run_name,
        num_agents,
        yaml_path,
        video_prefix,
        seed=0,
        max_num_cols=8,
        local_mode=False,
        steps=int(1e10),
        num_workers=5
):
    name_ckpt_mapping = get_name_ckpt_mapping(yaml_path, num_agents)

    assert isinstance(prediction, dict)
    assert isinstance(name_ckpt_mapping, dict)
    for key, val in prediction.items():
        assert key in name_ckpt_mapping
        assert isinstance(val, dict)

    assert env_name == "BipedalWalker-v2", "We only support BipedalWalker-v2 " \
                                           "currently!"

    gvr = GridVideoRecorder(
        video_path=video_prefix,
        env_name=env_name,
        run_name=run_name,
        local_mode=local_mode
    )

    new_name_ckpt_mapping, name_loc_mapping, name_row_mapping, \
    name_col_mapping = _transform_name_ckpt_mapping(
        name_ckpt_mapping, prediction, max_num_cols=max_num_cols
    )

    assert new_name_ckpt_mapping.keys() == name_row_mapping.keys(
    ) == name_loc_mapping.keys()

    frames_dict, extra_info_dict = gvr.generate_frames(
        new_name_ckpt_mapping,
        num_steps=steps,
        seed=seed,
        name_column_mapping=name_col_mapping,
        name_row_mapping=name_row_mapping,
        name_loc_mapping=name_loc_mapping,
        num_workers=num_workers
    )

    gvr.generate_video(frames_dict, extra_info_dict)

    gvr.close()


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

    name_row_mapping = {key: "TEST ROW" for key in name_ckpt_mapping.keys()}
    name_col_mapping = {key: "TEST COL" for key in name_ckpt_mapping.keys()}
    name_loc_mapping = {
        key: (int((idx + 1) / 3), int((idx + 1) % 3))
        for idx, key in enumerate(name_ckpt_mapping.keys())
    }

    frames_dict, extra_info_dict = gvr.generate_frames(
        name_ckpt_mapping, args.steps, args.iters, args.seed, name_col_mapping,
        name_row_mapping, name_loc_mapping
    )

    gvr.generate_video(frames_dict, extra_info_dict)

    gvr.close()
