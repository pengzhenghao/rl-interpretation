from __future__ import absolute_import, division, print_function, \
    absolute_import, division, print_function

from process_data.process_data import get_name_ckpt_mapping
from visualize.record_video import (create_parser, GridVideoRecorder,
                                    generate_grid_of_videos,rename_agent)
from test.utils import get_ppo_agent
from visualize.generate_gif import generate_gif_from_agent
from utils import initialize_ray
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080

FPS = 50


def test_genarate_gif():
    initialize_ray(test_mode=True)
    agent = get_ppo_agent("BipedalWalker-v2")
    ret = generate_gif_from_agent(agent, "test_agent", "/tmp/test_genrate_gif")
    print(ret)
    return ret


def test_cluster_video_generation():
    parser = create_parser()
    args = parser.parse_args()

    name_ckpt_mapping = get_name_ckpt_mapping(args.yaml, number=2)

    gvr = GridVideoRecorder(
        video_path=args.yaml[:-5], local_mode=args.local_mode, fps=FPS
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


def test_gif_generation():
    yaml = "yaml/test-2-agents.yaml"
    name_ckpt_mapping = get_name_ckpt_mapping(yaml, number=2)

    gvr = GridVideoRecorder(
        video_path="data/vis/gif/test-2-agents", local_mode=True, fps=FPS
    )

    frames_dict, extra_info_dict = gvr.generate_frames(name_ckpt_mapping)
    name_path_dict = gvr.generate_gif(frames_dict, extra_info_dict)
    print(name_path_dict)
    return name_path_dict
    # gvr.close()


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
