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
from utils import build_config, VideoRecorder, \
    restore_agent, initialize_ray
from env_wrapper import BipedalWalkerWrapper

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


def build_env_maker(seed):
    env = BipedalWalkerWrapper()
    env.seed(seed)
    return lambda: env


BUILD_ENV_MAKER = {"BipedalWalker-v2": build_env_maker}


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
            self,
            # run_name, env_maker, env_name,
            num_steps,
            num_iters,
            seed
    ):
        # self.run_name = run_name
        # self.env_maker = env_maker
        # self.env_name = env_name
        self.num_steps = num_steps
        self.num_iters = num_iters
        self.seed = seed

    def collect_frames(self, run_name, env_name, env_maker, config, ckpt):
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
        # env.seed(self.seed)

        result = rollout(agent, env, self.num_steps, require_frame=True)
        frames, extra_info = result['frames'], result['frame_extra_info']
        env.close()
        return frames, extra_info


class GridVideoRecorder(object):
    def __init__(
            self,
            video_path,
            # env_name, run_name,
            local_mode=False
    ):
        initialize_ray(local_mode)

        # single_env = gym.make(env_name)
        # self.env_name = env_name
        # self.run_name = run_name
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
                # self.run_name, ENVIRONMENT_MAPPING[self.env_name],
                # self.env_name,
                num_steps,
                num_iters,
                seed
            ) for _ in range(num_workers)
        ]

        for iteration in range(num_iteration):
            idx_start = iteration * num_workers
            idx_end = min((iteration + 1) * num_workers, num_agent)

            now = time.time()
            start = now
            object_id_dict = {}
            for incre, (name, ckpt_dict) in \
                    enumerate(name_ckpt_mapping_range[idx_start: idx_end]):
                assert isinstance(ckpt_dict, dict)
                ckpt = ckpt_dict["path"]
                run_name = ckpt_dict["run_name"]
                env_name = ckpt_dict["env_name"]
                env_maker = BUILD_ENV_MAKER[env_name](seed)
                config = build_config(ckpt, args_config)
                object_id_dict[name] = workers[incre].collect_frames.remote(
                    run_name, env_name, env_maker, config, ckpt
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

                del frames
                del extra_info

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

        unique_locations = set(locations)
        no_specify_location = len(unique_locations) == 1 and next(
            iter(unique_locations)
        ) is None
        assert len(unique_locations) == len(locations) or no_specify_location
        if no_specify_location:
            grids = len(frames_dict)
        else:
            max_row = max([row + 1 for row, _ in locations])
            max_col = max([col + 1 for _, col in locations])
            grids = {"col": max_col, "row": max_row}
        vr = VideoRecorder(self.video_path, grids=grids)
        path = vr.generate_video(frames_dict, extra_info_dict)
        return path

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
        name_ckpt_mapping, prediction, name_callback=None, max_num_cols=10
):
    # Function:
    # 1. re-order the OrderedDict
    # 2. add distance information in name
    clusters = set([c_info['cluster'] for c_info in prediction.values()])
    if name_callback is None:
        name_callback = lambda x: x
    assert callable(name_callback)
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

            new_name = name_callback(name)

            new_name_ckpt_mapping[new_name] = name_ckpt_mapping[name]
            name_loc_mapping[new_name] = loc
            if old_row_mapping is not None:
                name_row_mapping[new_name] = old_row_mapping[name]
            if old_col_mapping is not None:
                name_col_mapping[new_name] = old_col_mapping[name]

    return new_name_ckpt_mapping, name_loc_mapping, name_row_mapping, \
           name_col_mapping


def rename_agent(old_name, info=None):
    # old_name look like: <PPO seed=10 rew=10.1>
    # or for ablation study: <PPO seed=1 rew=29.35 policy/model/fc_out/unit34>
    # new name look like: <PPO,s10,r10.1> or <PPO,s1,r29.35,out34>
    components = old_name.split(" ")
    new_name = components[0]
    for com in components[1:]:
        if "=" in com:
            # We expect com to be like "seed=10" or "rew=201"
            # Then we transform it to "s10" or "r201"
            figure = eval(com.split('=')[1])
            new_name += "{}{:.0f}".format(com.split('=')[0][0], figure)
        else:
            layer_name, unit_name = com.split("/")[-2:]

            # layer_name should be like: fc_out or fc2
            assert layer_name.startswith("fc")
            layer_name = "o" if layer_name.endswith("out") else layer_name[2:]

            # unit_name should be like: no_ablation or unit32
            assert unit_name.startswith("unit") or unit_name.startswith("no")
            unit_name = unit_name[4:] if unit_name.startswith("unit") else "no"

            new_name += ",{}l{}".format(layer_name, unit_name)
    if info is not None and "distance" in info:
        new_name += ",d{:.2f}".format(info['distance'])
    return new_name


def generate_video_of_cluster(
        prediction,
        # env_name,
        # run_name,
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

    # assert env_name == "BipedalWalker-v2", \
    #     "We only support BipedalWalker-v2 currently!"

    new_name_ckpt_mapping, name_loc_mapping, name_row_mapping, \
    name_col_mapping = _transform_name_ckpt_mapping(
        name_ckpt_mapping, prediction, name_callback=rename_agent,
        max_num_cols=max_num_cols
    )

    assert new_name_ckpt_mapping.keys() == \
           name_row_mapping.keys() == name_loc_mapping.keys()
    generate_grid_of_videos(
        new_name_ckpt_mapping, video_prefix, name_row_mapping,
        name_col_mapping, name_loc_mapping, seed, None, local_mode, steps,
        num_workers
    )


def generate_grid_of_videos(
        name_ckpt_mapping,
        video_prefix,
        name_row_mapping=None,
        name_col_mapping=None,
        name_loc_mapping=None,
        seed=0,
        name_callback=rename_agent,
        local_mode=False,
        steps=int(1e10),
        num_workers=5
):
    if name_callback is not None:
        assert callable(name_callback)
        new_name_ckpt_mapping = OrderedDict()
        for old_name, val in name_ckpt_mapping.items():
            new_name = name_callback(old_name)
            new_name_ckpt_mapping[new_name] = val
        name_ckpt_mapping = new_name_ckpt_mapping
    gvr = GridVideoRecorder(
        video_path=video_prefix,
        # env_name=env_name,
        # run_name=run_name,
        local_mode=local_mode
    )
    frames_dict, extra_info_dict = gvr.generate_frames(
        name_ckpt_mapping,
        num_steps=steps,
        seed=seed,
        name_column_mapping=name_col_mapping,
        name_row_mapping=name_row_mapping,
        name_loc_mapping=name_loc_mapping,
        num_workers=num_workers
    )
    path = gvr.generate_video(frames_dict, extra_info_dict)
    gvr.close()
    print("Video has been saved at: <{}>".format(path))
    return path


def test_cluster_video_generation():

    parser = create_parser()
    args = parser.parse_args()

    name_ckpt_mapping = get_name_ckpt_mapping(args.yaml, number=2)

    gvr = GridVideoRecorder(
        video_path=args.yaml[:-5], local_mode=args.local_mode
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


def test_es_compatibility():
    name_ckpt_mapping = get_name_ckpt_mapping(
        "data/es-30-agents-0818.yaml", 10
    )
    generate_grid_of_videos(
        name_ckpt_mapping,
        "data/tmp_test_es_compatibility",
        steps=50,
        name_callback=rename_agent,
        num_workers=10
    )


if __name__ == "__main__":
    test_es_compatibility()
