from collections import OrderedDict

import cv2
import numpy as np

from toolbox.visualize.record_video import GridVideoRecorder


def get_velocity(extra_dict, agent_name, env_name):
    #     print("env_name:", env_name)
    if env_name.startswith("HalfCheetah"):
        return np.array(
            [t[0][8] for t in extra_dict['trajectory'][agent_name]]
        )
    elif env_name.startswith("Hopper-v3"):
        # Hopper has 6 DOF, first 5 is, then the 5 is ok.
        return np.array(
            [t[0][5] for t in extra_dict['trajectory'][agent_name]]
        )
    elif env_name.startswith("Walker"):
        return np.array(
            [t[0][8] for t in extra_dict['trajectory'][agent_name]]
        )


def remove_background(frame_list, require_close_operation=False):
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


def collect_frame(ckpt, agent_name, vis_env_name, num_steps=200, threshold=0):
    # assert os.path.exists(ckpt), ckpt
    output_path = "/tmp/tmp_{}_{}_{}_{}".format(
        agent_name, vis_env_name, num_steps, threshold
    )

    fps = 20
    gvr = GridVideoRecorder(
        video_path=output_path, fps=fps, require_full_frame=True
    )

    agent = SunAgentWrapper(ckpt, vis_env_name)
    agent.config['env'] = vis_env_name

    for i in range(4):
        frames_dict, extra_info_dict = gvr.generate_frames_from_agent(
            agent,
            agent_name,
            require_trajectory=True,
            num_steps=num_steps,
            seed=i * 100
        )

        reward = extra_info_dict['reward'][agent_name][-1]
        if reward >= threshold:
            break
        print(
            "Current reward {} have not exceed the threshold {}, so we have "
            "to add additional rollout.".format(reward, threshold)
        )

    velocity = get_velocity(extra_info_dict, agent_name, vis_env_name)
    frames_list = frames_dict[agent_name]['frames']
    print(
        "Start to remove background for ckpt {}. We have collect {} "
        "frames".format(ckpt, len(frames_list))
    )
    new_frame_list = remove_background(frames_list)
    return new_frame_list, velocity, extra_info_dict, frames_dict


def get_boundary(mask):
    bottom = np.argmax(mask.mean(1) != 0)
    top = bottom + np.argmin(mask.mean(1)[bottom:] != 0)
    left = np.argmax(mask.mean(0) != 0)
    right = left + np.argmin(mask.mean(0)[left:] != 0)
    return bottom, top, left, right


def draw_one_exp(
        frame_list,
        velocity,
        clip_top=200,
        alpha=0.5,
        skip_frame=5,
        velocity_multiplier=20,
        display=True,
        draw_last_frame=True,
        num_inter=2,
        resize=False
):
    alpha_dim = 3
    information = []

    # collect the basic message of frames
    for i, frame in enumerate(frame_list):
        frame = frame.copy()
        mask = frame.mean(2) > 10
        if_skip = i % skip_frame != 0

        if if_skip:
            frame[mask, alpha_dim] = int(255 * alpha)
        bottom, top, left, right = get_boundary(mask)
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
    frame_height = 500 - clip_top
    total_offset = sum([info['offset'] for info in information])
    x = 0
    canvas = np.ones(
        (500 - clip_top, x + total_offset + 500, 4), dtype=np.uint8
    ) * 255
    canvas[:, :, 3] = 0

    if num_inter >= 1:
        for i, (info, info2) in enumerate(zip(information[:-1],
                                              information[1:])):
            bottom, top, left, right = info['loc']
            width = info['width']
            offset = info['offset']
            mask = info['mask']
            frame = info['frame']
            shape = frame.shape
            if info['skip']:
                x_list = np.linspace(0, offset, num_inter).astype(int)
                bottom2 = info2['loc'][1]
                y_increase = bottom2 - top
                y_off_list = np.linspace(0, y_increase, num_inter).astype(int)
                for x_off, y_off in zip(x_list, y_off_list):
                    if y_off >= 0:
                        y1 = max(frame_height - shape[0] - y_off, 0)
                        new_mask = mask[:-y1].copy()
                        canvas[y1: - y_off, x + x_off: x + x_off + width][
                            new_mask] \
                            = frame[:-y1][new_mask]
                    else:
                        new_mask = mask[:-y1].copy()
                        canvas[y1: - y_off, x + x_off: x + x_off + width][
                            new_mask] \
                            = frame[:-y1][new_mask]

            x += offset
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

    if draw_last_frame:
        info = information[-1]
        width = info['width']
        mask = info['mask']
        frame = info['frame']
        shape = frame.shape
        frame[:, :, alpha_dim] = 255
        canvas[-shape[0]:, x:x + width][mask] = frame[mask]
    return canvas


DEFAULT_CONFIG = dict(
    start=0, interval=190, skip_frame=20, alpha=0.48, velocity_multiplier=7
)


def draw_all_result(result_dict, choose_index=None, config=None, threshold=0):
    config = config or DEFAULT_CONFIG
    index_ckpt_map = {}
    fig_dict = {}

    for i, (k, v) in enumerate(result_dict.items()):

        if (choose_index is not None) and (i not in choose_index):
            continue

        if len(v['frame']) > 800 and i > 10:
            continue

        start = config['start']
        interval = config['interval']
        skip_frame = config['skip_frame']
        alpha = config['alpha']
        velocity_multiplier = config['velocity_multiplier']

        new_frame_list = v['frame']
        velocity = v['velocity']

        reward = v['real_reward']
        if reward < threshold:
            print(
                "Reward is {} less than the threshold {}. So we skip agent "
                "with index {}".format(reward, threshold, i)
            )
            continue

        if interval is None:
            interval = len(new_frame_list) - start

        if len(new_frame_list) < start + interval:
            print(
                "The index {} agent has too small rollout length {}"
                " and can not achieve {} frames requirement.".format(
                    i, len(new_frame_list), start + interval
                )
            )
            continue

        index_ckpt_map[i] = k
        fig = draw_one_exp(
            new_frame_list[start:start + interval],
            velocity[start:start + interval],
            alpha=alpha,
            clip_top=100,
            skip_frame=skip_frame,
            velocity_multiplier=velocity_multiplier,
            display=True,
            draw_last_frame=False,
            num_inter=0
        )
        fig_dict[k] = fig
        print(
            "(LOOK UP) Current index: {}, Claim Reward: {}, Real Reward:"
            "{}\n\n".format(i, v['reward'], v['real_reward'])
        )

    return index_ckpt_map, fig_dict


def plot(
        index_ckpt_map,
        fig_dict,
        result_dict,
        choose=None,
        width=5000,
        clip=100,
        put_text=True
):
    if choose is None:
        choose = list(index_ckpt_map.keys())

    index_key_map = OrderedDict()
    for i in choose:
        index_key_map[i] = index_ckpt_map[i]

    frame = fig_dict[index_ckpt_map[choose[0]]]

    height = frame.shape[0] - clip
    canvas = np.zeros((height * len(choose), width, 4), dtype=np.uint8)

    for i, ckpt in enumerate(index_key_map.values()):

        reward = result_dict[ckpt]['reward']

        frame = fig_dict[ckpt]

        title = "Reward: {:.2f}".format(reward)

        clip_width = min(frame.shape[1], width)

        if put_text:
            frame_return = cv2.putText(
                frame[clip:, :clip_width].copy(), title, (20, height - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0, 255), 2
            )
        else:
            frame_return = frame[clip:, :clip_width].copy()

        canvas[i * height:(i + 1) * height, :clip_width, ...] = frame_return

    return canvas
