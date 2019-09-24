from __future__ import absolute_import, division, print_function, \
    absolute_import, division, print_function

# import sys
# sys.path.append("../")
from toolbox.process_data.process_data import get_name_ckpt_mapping, read_yaml
from toolbox.test.utils import get_ppo_agent
from toolbox.utils import initialize_ray
from toolbox.visualize.generate_gif import generate_gif_from_agent
from toolbox.visualize.record_video import (
    create_parser, GridVideoRecorder, generate_grid_of_videos, rename_agent
)
from toolbox.evaluate.evaluate_utils import restore_agent

VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080

FPS = 50


def test_generate_gif_from_agent():
    initialize_ray(test_mode=True)
    agent = get_ppo_agent("BipedalWalker-v2")
    ret = generate_gif_from_agent(agent, "test_agent", "/tmp/test_genrate_gif")
    print(ret)
    return ret


def test_generate_gif_from_agent_mujoco_environemnt():
    initialize_ray(test_mode=True)
    agent = get_ppo_agent("HalfCheetah-v2")
    output_path = "(delete-me!)test_genrate_gif_mujoco"
    agent_name = "test_agent_mujoco"

    gvr = GridVideoRecorder(
        video_path=output_path, fps=FPS, require_full_frame=True
    )
    frames_dict, extra_info_dict = gvr.generate_frames_from_agent(
        agent, agent_name
    )

    name_path_dict = gvr.generate_gif(frames_dict, extra_info_dict)
    print("Gif has been saved at: ", name_path_dict)


def test_generate_gif_from_restored_agent_mujoco_environemnt():
    initialize_ray(test_mode=True)
    # agent = get_ppo_agent("HalfCheetah-v2")

    ckpt = "/home/zhpeng/ray_results/0915-hc-ppo-5-agents/" \
           "PPO_HalfCheetah-v2_2_seed=2_2019-09-15_15-01-01hyqn2x2v/" \
           "checkpoint_1060/checkpoint-1060"

    agent = restore_agent("PPO", ckpt, "HalfCheetah-v2")

    output_path = "(delete-me!)test_genrate_gif_mujoco"
    agent_name = "test_agent_mujoco"

    gvr = GridVideoRecorder(
        video_path=output_path, fps=FPS, require_full_frame=True
    )
    frames_dict, extra_info_dict = gvr.generate_frames_from_agent(
        agent, agent_name
    )

    name_path_dict = gvr.generate_gif(frames_dict, extra_info_dict)
    print("Gif has been saved at: ", name_path_dict)


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


def test_generate_single_video():
    name_ckpt_mapping = read_yaml("data/yaml/test-2-agents.yaml", 1)
    path = generate_grid_of_videos(
        name_ckpt_mapping,
        "/tmp/test_single_agent",
        name_callback=lambda x, y=None: x,
        require_full_frame=True,
        require_text=False
    )

    print("test finish: ", path)


def test_generate_two_videos():
    from toolbox.process_data.process_data import read_batch_yaml
    """You should enter this function at the root dir, s.t. root/toolbox/.."""
    yaml_path_dict_list = [
        {
            "number": 1,
            "mode": "top",
            "path": "data/yaml/ppo-300-agents.yaml"
            # "path": "data/yaml/test-2-agents.yaml"
        },
        {
            "number": 1,
            "mode": "top",
            "path": "data/yaml/es-30-agents-0818.yaml"
        }
    ]

    name_ckpt_mapping = read_batch_yaml(yaml_path_dict_list)
    path = generate_grid_of_videos(
        name_ckpt_mapping,
        # "/tmp/test_two_agents",
        "./video_0916/video_double_agent",
        name_callback=lambda x, y=None: x,
        require_full_frame=True,
        require_text=False
    )

    print("test generating two videos finished: ", path)


def test_generate_two_videos2():
    name_ckpt_mapping = read_yaml("data/yaml/test-2-agents.yaml", 2)
    path = generate_grid_of_videos(
        name_ckpt_mapping,
        "/tmp/test_double_agent",
        name_callback=lambda x, y=None: x,
        require_full_frame=True,
        require_text=False,
    )

    print("test finish: ", path)


if __name__ == '__main__':
    import os

    # os.chdir("../../")
    print("CURRENT LOCATION: ", os.getcwd())
    # test_generate_two_videos()
    # test_generate_two_videos2()
    # test_generate_gif_from_agent_mujoco_environemnt()
    test_generate_gif_from_restored_agent_mujoco_environemnt()
