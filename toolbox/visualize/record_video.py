from __future__ import absolute_import, division, print_function, \
    absolute_import, division, print_function

import argparse
import copy
import json
import time
from collections import OrderedDict

import numpy as np
import ray

from toolbox.env.env_maker import get_env_maker
from toolbox.evaluate import build_config, restore_agent
from toolbox.evaluate.rollout import rollout
from toolbox.process_data.process_data import get_name_ckpt_mapping, read_yaml
from toolbox.represent.process_fft import get_period
from toolbox.utils import initialize_ray
from toolbox.visualize.visualize_utils import VideoRecorder

VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080

FPS = 50

# info_datails should contain:
#   default_value: the default value of this measurement
#   text_function: a function transform the value to string
#   pos_ratio: a tuple: (left_ratio, bottom_ratio)
PRESET_INFORMATION_DICT = {
    "reward": {
        "default_value": 0.0,
        "text_function": lambda val: "Rew {:07.2f}".format(val),
        "pos_ratio": (0.95, 0.9)
    },
    "step": {
        "default_value": 0,
        "text_function": lambda val: "Step {}".format(val),
        "pos_ratio": (0.95, 0.8)
    },
    "done": {
        "default_value": False,
        "text_function": lambda val: "X" if val else "",
        "pos_ratio": (0.07, 0.9)
    },
    "value_function": {
        "default_value": 0.0,
        "text_function": lambda val: "Val {:.3f}".format(val),
        "pos_ratio": (0.95, 0.7)
    },
    "title": {
        "default_value": "",
        "text_function": lambda val: val,
        "pos_ratio": (0.95, 0.05)
    }
}

from toolbox.abstract_worker import WorkerBase, WorkerManagerBase


class _CollectFramesWorker(WorkerBase):
    def collect_frames(
            self,
            num_steps,
            run_name,
            env_name,
            config,
            ckpt,
            require_full_frame=False,
            render_mode="rgb_array",
            ideal_steps=None,
            random_seed=False
    ):
        agent = restore_agent(run_name, ckpt, env_name, config)
        # if ideal_steps is not None:
        tmp_frames = []
        tmp_extra_info = []

        # We allow 10 attemps.
        for i in range(10):
            if random_seed:
                seed = np.random.randint(0, 10000)
            else:
                seed = i
            env_maker = get_env_maker(env_name, require_render=True)
            env = env_maker(seed=seed)
            result = rollout(
                agent,
                env,
                env_name,
                num_steps,
                require_frame=True,
                require_full_frame=require_full_frame,
                render_mode=render_mode
            )
            frames, extra_info = result['frames'], result['frame_extra_info']

            if len(frames) > len(tmp_frames):
                tmp_frames = copy.deepcopy(frames)
                tmp_extra_info = copy.deepcopy(extra_info)

            if (ideal_steps is None) or (len(frames) > ideal_steps):
                frames = tmp_frames
                extra_info = tmp_extra_info
                break
            else:
                print(
                    "In collect_frames, current frame length is {} and "
                    "we expect length {}. So we rerun the rollout "
                    "with different seed {}."
                    " Current length of potential 'frames' is {}".format(
                        len(frames), ideal_steps, i + 1, len(tmp_frames)
                    )
                )
        env.close()
        agent.stop()
        return frames, extra_info


class CollectFramesManager(WorkerManagerBase):
    def __init__(
            self,
            num_steps,
            require_full_frame,
            num_workers,
            total_num=None,
            log_interval=1,
    ):
        super(CollectFramesManager, self).__init__(
            num_workers, _CollectFramesWorker, total_num, log_interval,
            "collect frame"
        )
        self.num_steps = num_steps
        self.require_full_frame = require_full_frame

    def collect_frames(
            self, index, run_name, env_name, config, ckpt, render_mode,
            ideal_steps, random_seed
    ):
        self.submit(
            index, self.num_steps, run_name, env_name, config, ckpt,
            self.require_full_frame, render_mode, ideal_steps, random_seed
        )


class GridVideoRecorder(object):
    def __init__(
            self, video_path, local_mode=False, fps=50,
            require_full_frame=False
    ):
        initialize_ray(local_mode)
        self.video_path = video_path
        self.fps = fps
        self.require_full_frame = require_full_frame

    def generate_frames_from_agent(
            self,
            agent,
            agent_name,
            num_steps=None,
            seed=0,
            render_mode="rgb_array",
            require_trajectory=False,
            ideal_steps=None
    ):
        config = agent.config
        env_name = config["env"]
        env = get_env_maker(env_name, require_render=True)()
        if seed is not None:
            assert isinstance(seed, int)
            env.seed(seed)

        for iteration in range(10):

            result = copy.deepcopy(
                rollout(
                    agent,
                    env,
                    env_name,
                    num_steps,
                    require_frame=True,
                    require_trajectory=require_trajectory,
                    require_full_frame=self.require_full_frame,
                    render_mode=render_mode
                )
            )
            frames, extra_info = result['frames'], result['frame_extra_info']
            if require_trajectory:
                extra_info['trajectory'] = result['trajectory']

            if ideal_steps is None:
                break
            elif len(frames) > ideal_steps:
                break

        env.close()
        agent.stop()
        period_info = extra_info['period_info']
        if period_info:
            period_source = np.stack(period_info)
            period = get_period(period_source, self.fps)
            print(
                "period for agent <{}> is {}, its len is {}".format(
                    agent_name, period, len(frames)
                )
            )
        else:
            period = 100
        frames_info = {
            "frames": frames,
            "column": None,
            "row": None,
            "loc": None,
            "period": period
        }
        return_dict = {agent_name: frames_info}
        extra_info_dict = PRESET_INFORMATION_DICT.copy()
        for key, val in extra_info.items():
            if key in extra_info_dict:
                extra_info_dict[key][agent_name] = val
            elif key == "vf_preds":
                extra_info_dict["value_function"][agent_name] = val
            elif key == "trajectory" and require_trajectory:
                if "trajectory" in extra_info_dict:
                    extra_info_dict["trajectory"][agent_name] = val
                else:
                    extra_info_dict["trajectory"] = {agent_name: val}
        extra_info_dict['title'][agent_name] = agent_name

        new_extra_info_dict = PRESET_INFORMATION_DICT.copy()
        for key in PRESET_INFORMATION_DICT.keys():
            new_extra_info_dict[key].update(extra_info_dict[key])

        if require_trajectory:
            new_extra_info_dict["trajectory"] = extra_info_dict["trajectory"]

        new_extra_info_dict['frame_info'] = {
            "width": frames[0].shape[1],
            "height": frames[0].shape[0]
        }

        return return_dict, new_extra_info_dict

    def generate_frames(
            self,
            name_ckpt_mapping,
            num_steps=None,
            num_iters=1,
            seed=0,
            name_column_mapping=None,
            name_row_mapping=None,
            name_loc_mapping=None,
            args_config=None,
            num_workers=10,
            render_mode="rgb_array",
            ideal_steps=None,
            random_seed=False
    ):
        assert isinstance(name_ckpt_mapping, OrderedDict), \
            "The name-checkpoint dict is not OrderedDict!!! " \
            "We suggest you to use OrderedDict."
        name_ckpt_mapping_range = list(name_ckpt_mapping.items())

        frames_dict = {}
        extra_info_dict = PRESET_INFORMATION_DICT.copy()

        collect_manager = CollectFramesManager(
            num_steps, self.require_full_frame, num_workers,
            len(name_ckpt_mapping_range)
        )

        for incre, (name, ckpt_dict) in enumerate(name_ckpt_mapping):
            assert isinstance(ckpt_dict, dict)
            ckpt = ckpt_dict["path"]
            run_name = ckpt_dict["run_name"]
            env_name = ckpt_dict["env_name"]
            # env_maker = get_env_maker(env_name, require_render=True)
            is_es_agent = run_name == "ES"
            config = build_config(ckpt, args_config, is_es_agent)

            collect_manager.collect_frames(
                name, run_name, env_name, config, ckpt, render_mode,
                ideal_steps, random_seed
            )

        collect_results = collect_manager.get_result()

        for name, (new_frames, new_extra_info) in collect_results.items():

            period_info = new_extra_info['period_info']
            if period_info:
                period_source = np.stack(period_info)
                period = get_period(period_source, self.fps)
                print(
                    "period for agent <{}> is {}, its len is {}".format(
                        name, period, len(new_frames)
                    )
                )
            else:
                period = 100
            frames_info = {
                "frames":
                new_frames,
                "column":
                None
                if name_column_mapping is None else name_column_mapping[name],
                "row":
                None if name_row_mapping is None else name_row_mapping[name],
                "loc":
                None if name_loc_mapping is None else name_loc_mapping[name],
                "period":
                period
            }

            frames_dict[name] = frames_info
            for key, val in new_extra_info.items():
                if key in extra_info_dict:
                    extra_info_dict[key][name] = val
                elif key == "vf_preds":
                    extra_info_dict["value_function"][name] = val
            extra_info_dict['title'][name] = name

        new_extra_info_dict = PRESET_INFORMATION_DICT.copy()
        for key in PRESET_INFORMATION_DICT.keys():
            new_extra_info_dict[key].update(extra_info_dict[key])
        new_extra_info_dict['frame_info'] = {
            "width": new_frames[0].shape[1],
            "height": new_frames[0].shape[0]
        }
        return frames_dict, new_extra_info_dict

    def generate_video(
            self,
            frames_dict,
            extra_info_dict,
            require_text=True,
            test_mode=False,
            four_lines=False
    ):
        print(
            "Start generating grid containing {} videos.".format(
                len(frames_dict)
            )
        )
        locations = [f_info['loc'] for f_info in frames_dict.values()]

        unique_locations = set(locations)
        no_specify_location = len(unique_locations) == 1 and (
            next(iter(unique_locations)) is None
        )
        assert len(unique_locations) == len(locations) or no_specify_location
        if no_specify_location:
            grids = len(frames_dict)
        else:
            max_row = max([row + 1 for row, _ in locations])
            max_col = max([col + 1 for _, col in locations])
            grids = {"col": max_col, "row": max_row}
        vr = VideoRecorder(
            self.video_path,
            grids=grids,
            fps=self.fps,
            test_mode=test_mode,
            four_lines=four_lines
        )
        path = vr.generate_video(frames_dict, extra_info_dict, require_text)
        # if we are at test_mode, then 'path' is a frame.
        return path

    def generate_single_video(self, frames_dict, *args, **kwargs):
        vr = VideoRecorder(self.video_path, fps=self.fps)
        path = vr.generate_single_video(frames_dict)
        return path

    def generate_gif(self, frames_dict, extra_info_dict):
        print(
            "Start generating grid containing {} videos.".format(
                len(frames_dict)
            )
        )
        # path = osp.join(self.video_path, "beginning")
        vr = VideoRecorder(
            self.video_path, generate_gif=True, fps=self.fps, scale=0.5
        )
        name_path_dict = vr.generate_video(frames_dict, extra_info_dict)
        return name_path_dict

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
        num_agents,
        yaml_path,
        video_prefix,
        seed=0,
        max_num_cols=8,
        local_mode=False,
        steps=int(1e10),
        num_workers=5,
        require_text=True
):
    name_ckpt_mapping = get_name_ckpt_mapping(yaml_path, num_agents)

    assert isinstance(prediction, dict)
    assert isinstance(name_ckpt_mapping, dict)
    for key, val in prediction.items():
        assert key in name_ckpt_mapping
        assert isinstance(val, dict)

    new_name_ckpt_mapping, name_loc_mapping, name_row_mapping, \
    name_col_mapping = _transform_name_ckpt_mapping(
        name_ckpt_mapping, prediction, name_callback=rename_agent,
        max_num_cols=max_num_cols
    )

    assert new_name_ckpt_mapping.keys() == \
           name_row_mapping.keys() == name_loc_mapping.keys()
    generate_grid_of_videos(
        new_name_ckpt_mapping,
        video_prefix,
        name_row_mapping,
        name_col_mapping,
        name_loc_mapping,
        seed,
        name_callback=None,
        local_mode=local_mode,
        steps=steps,
        num_workers=num_workers,
        require_text=require_text
    )


def generate_grid_of_videos(
        name_ckpt_mapping,
        video_prefix,
        name_row_mapping=None,
        name_col_mapping=None,
        name_loc_mapping=None,
        seed=0,
        name_callback=rename_agent,
        require_full_frame=False,
        require_text=True,
        local_mode=False,
        steps=int(1e10),
        num_workers=5,
        fps=None,
        test_mode=False,
        rerun_if_steps_is_not_enough=False,
        four_lines=False
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
        local_mode=local_mode,
        fps=fps or FPS,
        require_full_frame=require_full_frame
    )
    if test_mode:
        steps = 1
    frames_dict, extra_info_dict = gvr.generate_frames(
        name_ckpt_mapping,
        num_steps=steps,
        seed=seed,
        name_column_mapping=name_col_mapping,
        name_row_mapping=name_row_mapping,
        name_loc_mapping=name_loc_mapping,
        num_workers=num_workers,
        ideal_steps=steps if rerun_if_steps_is_not_enough else None,
        random_seed=rerun_if_steps_is_not_enough
    )
    path = gvr.generate_video(
        frames_dict,
        extra_info_dict,
        require_text,
        test_mode=test_mode,
        four_lines=four_lines
    )
    gvr.close()
    print("Video has been saved at: <{}>".format(path))
    return path


def generate_single_video(yaml_path, output_path):
    assert yaml_path.endswith(".yaml")
    name_ckpt_mapping = read_yaml(yaml_path, 1)
    path = generate_grid_of_videos(
        name_ckpt_mapping,
        output_path,
        name_callback=lambda x, y=None: x,
        require_full_frame=True,
        require_text=False
    )
    print("Successfully generated video at: ", path)
    return path


def generate_gif(yaml_path, output_dir):
    name_ckpt_mapping = get_name_ckpt_mapping(yaml_path)
    gvr = GridVideoRecorder(output_dir, fps=FPS)
    frames_dict, extra_info_dict = gvr.generate_frames(name_ckpt_mapping)
    name_path_dict = gvr.generate_gif(frames_dict, extra_info_dict)
    return name_path_dict


def create_parser():
    parser = argparse.ArgumentParser(
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


if __name__ == "__main__":
    pass
