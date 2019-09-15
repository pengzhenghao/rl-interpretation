import sys
sys.path.append("../")
from toolbox.process_data.process_data import get_name_ckpt_mapping
from toolbox.visualize.record_video import GridVideoRecorder

FPS = 50


def generate_gif_from_agent(agent, agent_name, output_path):
    gvr = GridVideoRecorder(video_path=output_path, fps=FPS)
    frames_dict, extra_info_dict = gvr.generate_frames_from_agent(
        agent, agent_name
    )
    name_path_dict = gvr.generate_gif(frames_dict, extra_info_dict)
    print("Gif has been saved at: ", name_path_dict)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--yaml-path", required=True, type=str)
    parser.add_argument("--output-path", required=True, type=str)
    args = parser.parse_args()
    # assert isinstance(args.exp_names, list) or isinstance(args.exp_names,
    # str)
    # assert args.output_path.endswith("yaml")

    name_ckpt_mapping = get_name_ckpt_mapping(args.yaml_path)

    gvr = GridVideoRecorder(video_path=args.output_path, fps=FPS)

    frames_dict, extra_info_dict = gvr.generate_frames(name_ckpt_mapping)
    name_path_dict = gvr.generate_gif(frames_dict, extra_info_dict)
    print(name_path_dict)
