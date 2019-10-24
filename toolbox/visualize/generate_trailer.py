from toolbox.process_data.process_data import get_name_ckpt_mapping
from toolbox.visualize.record_video import GridVideoRecorder

def generate_gif_from_agent(
        agent, agent_name, output_path, require_full_frame=False, _steps=None
):

    env_name = agent.config['env']
    fps = 50 if env_name.startswith("BipedalWalker") else 20

    gvr = GridVideoRecorder(
        video_path=output_path, fps=fps, require_full_frame=require_full_frame
    )
    frames_dict, extra_info_dict = gvr.generate_frames_from_agent(
        agent, agent_name, num_steps=_steps
    )

    name_path_dict = gvr.generate_gif(frames_dict, extra_info_dict)
    print("Gif has been saved at: ", name_path_dict)
    return name_path_dict
