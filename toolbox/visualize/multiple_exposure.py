import cv2
import numpy as np

from toolbox.env.env_maker import ENV_MAKER_LOOKUP
from toolbox.visualize.record_video import GridVideoRecorder

VELOCITY_RETRIEVE_LIST = [
    "HalfCheetah-v2",
    "HalfCheetah-v2-shadow",
    "HalfCheetah-v3",
    "HalfCheetah-v3-shadow",
    "Hopper-v3",
    "Hopper-v3-shadow",
    "Walker2d-v3",
    "Walker2d-v3-shadow",
]

halfcheetah_config = dict(
    start=0, interval=300, skip_frame=20, alpha=0.25, velocity_multiplier=7
)

walker_config = dict(
    start=0, interval=None, skip_frame=30, alpha=0.25, velocity_multiplier=10
)

hopper_config = dict(
    start=0, interval=None, skip_frame=30, alpha=0.48, velocity_multiplier=10
)

ENV_RENDER_CONFIG_LOOKUP = {
    "HalfCheetah-v2": halfcheetah_config,
    "HalfCheetah-v2-shadow": halfcheetah_config,
    "HalfCheetah-v3": halfcheetah_config,
    "HalfCheetah-v3-shadow": halfcheetah_config,
    "Hopper-v3": hopper_config,
    "Hopper-v3-shadow": hopper_config,
    "Walker2d-v3": walker_config,
    "Walker2d-v3-shadow": walker_config,
}


def _get_velocity(extra_dict, agent_name, env_name):
    assert env_name in ENV_MAKER_LOOKUP.keys()
    assert env_name in VELOCITY_RETRIEVE_LIST
    if env_name.startswith("HalfCheetah"):
        return np.array(
            [t[0][8] for t in extra_dict['trajectory'][agent_name]]
        )
    elif env_name.startswith("Hopper-v3"):
        return np.array(
            [t[0][5] for t in extra_dict['trajectory'][agent_name]]
        )
    elif env_name.startswith("Walker"):
        return np.array(
            [t[0][8] for t in extra_dict['trajectory'][agent_name]]
        )


def _remove_background(frame_list, require_close_operation=False):
    """
    This function remove the background of render image of Mujoco environment
    The input is a single frame array whose shape is [500, 500, 3]

    We require the frame belonging to the relatively later peroid of a rollout,
    since at that time the background is pure gray gradients.

    The output is a RGBA image array whose shape should be [500, 500, 4]
    """
    new_frame_list = []
    for frame in frame_list:
        frame = frame.copy()
        background = np.tile(frame[:, :1, :], [frame.shape[1], 1])
        mask = cv2.absdiff(frame, background)
        if require_close_operation:
            mask = cv2.erode(
                mask, kernel=np.ones((3, 3), np.uint8), iterations=2
            )
            mask = cv2.dilate(
                mask, kernel=np.ones((5, 5), np.uint8), iterations=1
            )
        mask = mask.mean(2) < 10
        mask = np.tile(mask[:, :, None], [1, 1, 4]).astype(np.uint8) * 255
        new_mask = mask == 255
        new_frame = frame.copy()
        new_frame = cv2.cvtColor(new_frame[..., ::-1], cv2.COLOR_RGB2RGBA)
        new_frame[new_mask] = 0
        new_frame_list.append(new_frame)

    return new_frame_list.copy()


def _get_boundary(mask):
    bottom = np.argmax(mask.mean(1) != 0)
    top = bottom + np.argmin(mask.mean(1)[bottom:] != 0)
    left = np.argmax(mask.mean(0) != 0)
    right = left + np.argmin(mask.mean(0)[left:] != 0)
    return bottom, top, left, right


def collect_frame(
        agent, agent_name, vis_env_name, num_steps=None, reward_threshold=None
):
    output_path = "/tmp/tmp_{}_{}_{}_{}".format(
        agent_name, vis_env_name, num_steps or "inf-steps", reward_threshold
        or "-inf"
    )  # temporary output_path and make no different.

    if reward_threshold is None:
        reward_threshold = float("-inf")

    fps = 20
    gvr = GridVideoRecorder(
        video_path=output_path, fps=fps, require_full_frame=True
    )

    for i in range(4):
        frames_dict, extra_info_dict = gvr.generate_frames_from_agent(
            agent,
            agent_name,
            require_trajectory=True,
            num_steps=num_steps,
            seed=i * 100
        )

        reward = extra_info_dict['reward'][agent_name][-1]
        if reward >= reward_threshold:
            break
        print(
            "Current reward {} have not exceed the reward_threshold {}, "
            "so we have "
            "to add additional rollout.".format(reward, reward_threshold)
        )

    velocity = _get_velocity(extra_info_dict, agent_name, vis_env_name)
    frames_list = frames_dict[agent_name]['frames']
    new_frame_list = _remove_background(frames_list)
    return new_frame_list, velocity, extra_info_dict, frames_dict


DEFAULT_CONFIG = dict(
    start=0,
    interval=100,
    skip_frame=20,
    alpha=0.48,
    velocity_multiplier=10,
    clip_top=0,
    draw_last_frame=False
)


def draw_one_exp(frame_list, velocity, draw_config=None):
    config = DEFAULT_CONFIG.copy()
    config.update(draw_config or {})

    start = config['start']
    interval = config['interval']
    skip_frame = config['skip_frame']
    alpha = config['alpha']
    velocity_multiplier = config['velocity_multiplier']
    clip_top = config['clip_top']
    draw_last_frame = config['draw_last_frame']

    if interval is None:
        interval = len(frame_list) - start

    if start + interval > len(frame_list):
        draw_frame_list = frame_list.copy()
    else:
        draw_frame_list = frame_list[start:start + interval].copy()

    if velocity is None:
        velocity = np.ones((len(draw_frame_list), ))

    alpha_dim = 3
    information = []

    # collect the basic message of frames
    for i, frame in enumerate(draw_frame_list):
        frame = frame.copy()
        mask = frame.mean(2) > 10
        if_skip = i % skip_frame != 0

        if if_skip:
            frame[mask, alpha_dim] = int(255 * alpha)
        bottom, top, left, right = _get_boundary(mask)
        mask = np.tile(mask[:, :, None], [1, 1, 4])
        information.append(
            {
                "mask": mask[clip_top:, left:right, ...].copy(),
                "frame": frame[clip_top:, left:right, ...].copy(),
                "loc": (bottom, top, left, right),
                "height": top - bottom,
                "width": right - left,
                "offset": int(max(velocity[i] * velocity_multiplier, 0)),
                "skip": if_skip
            }
        )

    # build canvas
    total_offset = sum([info['offset'] for info in information])
    x = 0
    canvas = np.ones(
        (500 - clip_top, x + total_offset + 500, 4), dtype=np.uint8
    ) * 255
    canvas[:, :, 3] = 0

    # draw the not highlighted frames
    x = 0
    for i, info in enumerate(information):
        width = info['width']
        offset = info['offset']
        mask = info['mask']
        frame = info['frame']
        shape = frame.shape
        if info['skip']:
            canvas[-shape[0]:, x:x + width][mask] = frame[mask]
        x += offset

    # draw the highlighted frames
    x = 0
    for i, info in enumerate(information):
        width = info['width']
        offset = info['offset']
        mask = info['mask']
        frame = info['frame']
        shape = frame.shape
        if not info['skip']:
            canvas[-shape[0]:, x:x + width][mask] = frame[mask]
        x += offset

    # draw the last frame if necessary
    if draw_last_frame:
        info = information[-1]
        width = info['width']
        mask = info['mask']
        frame = info['frame']
        shape = frame.shape
        frame[:, :, alpha_dim] = 255
        canvas[-shape[0]:, x:x + width][mask] = frame[mask]
    return canvas


# INTERFACE
def generate_multiple_exposure(
        agent,
        agent_name,
        output_path=None,
        vis_env_name=None,
        num_steps=None,
        reward_threshold=None,
        render_config=None,
        put_text=True
):
    env_name = agent.config['env']
    if (vis_env_name is not None) and (vis_env_name != env_name):
        agent.config['env'] = vis_env_name
        env_name = vis_env_name

    new_frame_list, velocity, extra_info_dict, frames_dict = \
        collect_frame(agent, agent_name, env_name, num_steps=num_steps,
                      reward_threshold=reward_threshold)

    default_config = ENV_RENDER_CONFIG_LOOKUP[env_name].copy()
    default_config.update(render_config or {})

    canvas = draw_one_exp(
        frame_list=new_frame_list, velocity=velocity, config=default_config
    )

    if put_text:
        real_reward = extra_info_dict['reward'][agent_name][-1]
        title = "{}, Episode Reward: {:.2f}".format(agent_name, real_reward)
        canvas = cv2.putText(
            canvas, title, (20, canvas.shape[0] - 2), cv2.FONT_HERSHEY_SIMPLEX,
            2.2, (0, 0, 0, 255), 2
        )

    alpha = np.tile(canvas[..., 3:], [1, 1, 3]) / 255  # in range [0, 1]
    white = np.ones_like(alpha) * 255
    return_canvas = np.multiply(canvas[..., :3], alpha).astype(np.uint8) + \
                    np.multiply(white, 1 - alpha).astype(np.uint8)
    return_canvas = cv2.cvtColor(return_canvas, cv2.COLOR_BGR2RGB)

    if output_path is not None:
        cv2.imwrite(output_path, return_canvas)

    return return_canvas
