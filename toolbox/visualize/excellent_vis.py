import csv
import os
import re
from collections import OrderedDict

import cv2
import numpy as np
import torch

# from toolbox.env.env_maker import get_env_maker
from toolbox.env.env_maker import get_env_maker
from toolbox.evaluate.sunhao_wrapper import SunAgentWrapper, args, conf, \
    ActorCritic
from toolbox.visualize.record_video import GridVideoRecorder


# %matplotlib inline
# np.random.seed(0)


def get_velocity(extra_dict, agent_name, env_name):
    #     print("env_name:", env_name)
    if env_name.startswith("HalfCheetah"):
        return np.array(
            [t[0][8] for t in extra_dict['trajectory'][agent_name]])
    elif env_name.startswith("Hopper-v3"):
        # Hopper has 6 DOF, first 5 is, then the 5 is ok.
        return np.array(
            [t[0][5] for t in extra_dict['trajectory'][agent_name]])
    elif env_name.startswith("Walker"):
        return np.array(
            [t[0][8] for t in extra_dict['trajectory'][agent_name]])


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
            mask = cv2.erode(mask, kernel=np.ones((3, 3), np.uint8),
                             iterations=2)
            mask = cv2.dilate(mask, kernel=np.ones((5, 5), np.uint8),
                              iterations=1)
        mask = mask.mean(2) < 10
        mask = np.tile(mask[:, :, None], [1, 1, 4]).astype(np.uint8) * 255
        new_mask = mask == 255
        new_frame = frame.copy()
        new_frame = cv2.cvtColor(new_frame[..., ::-1], cv2.COLOR_RGB2RGBA)
        new_frame[new_mask] = 0
        new_frame_list.append(new_frame)

    return new_frame_list.copy()


def collect_frame(ckpt, agent_name, vis_env_name, num_steps=200, threshold=0):
    assert os.path.exists(ckpt), ckpt

    output_path = "/tmp/SUPPORT_SUNHAO_0924"
    # env_name = ENV_NAME
    # fps = 50 if env_name.startswith("BipedalWalker") else 20
    fps = 20
    gvr = GridVideoRecorder(video_path=output_path, fps=fps,
                            require_full_frame=True)

    env = get_env_maker(vis_env_name)()
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    num_units = eval(re.search(r"_(\d+)hidden", ckpt).group(1))

    network = ActorCritic(num_inputs, num_actions, config=conf(num_units, 0.0),
                          layer_norm=args.layer_norm)
    network.load_state_dict(torch.load(ckpt))
    agent = SunAgentWrapper(network)
    agent.config['env'] = vis_env_name

    reward = threshold - 1

    for i in range(4):
        frames_dict, extra_info_dict = gvr.generate_frames_from_agent(
            agent, agent_name, require_trajectory=True, num_steps=num_steps,
            seed=i * 100
        )

        reward = extra_info_dict['reward'][agent_name][-1]
        if reward >= threshold:
            break
        print(
            "Current reward {} have not exceed the threshold {}, so we have "
            "to add additional rollout.".format(
                reward, threshold
            ))

    velocity = get_velocity(extra_info_dict, agent_name, vis_env_name)
    frames_list = frames_dict[agent_name]['frames']
    print(
        "Start to remove background for ckpt {}. We have collect {} "
        "frames".format(
            ckpt, len(frames_list)))
    new_frame_list = remove_background(frames_list)
    return new_frame_list, velocity, extra_info_dict, frames_dict


def get_boundary(mask):
    bottom = np.argmax(mask.mean(1) != 0)
    top = bottom + np.argmin(mask.mean(1)[bottom:] != 0)
    left = np.argmax(mask.mean(0) != 0)
    right = left + np.argmin(mask.mean(0)[left:] != 0)
    #     print()
    return bottom, top, left, right


def draw_one_exp(frame_list, velocity, clip_top=200, alpha=0.5, skip_frame=5,
                 velocity_multiplier=20,
                 display=True, draw_last_frame=True, num_inter=2,
                 resize=False):
    #     print("We are accepting {} frames.".format(len(frame_list)))
    alpha_dim = 3
    information = []
    for i, frame in enumerate(frame_list):
        frame = frame.copy()
        mask = frame.mean(2) > 10
        if_skip = i % skip_frame != 0

        if if_skip:
            frame[mask, alpha_dim] = int(255 * alpha)
        bottom, top, left, right = get_boundary(mask)
        mask = np.tile(mask[:, :, None], [1, 1, 4])
        information.append({
            "mask": mask[clip_top:, left: right, ...].copy(),
            "frame": frame[clip_top:, left: right, ...].copy(),
            "loc": (bottom, top, left, right),
            "height": top - bottom,
            "width": right - left,
            "offset": int(max(velocity[i] * velocity_multiplier, 0)),
            "skip": if_skip
        })
    frame_height = 500 - clip_top
    width = sum([info['width'] for info in information])
    total_offset = sum([info['offset'] for info in information])
    x = 0
    y = 100
    canvas = np.ones((500 - clip_top, x + total_offset + 500, 4),
                     dtype=np.uint8) * 255
    canvas[:, :, 3] = 0

    if num_inter >= 1:
        for i, (info, info2) in enumerate(
                zip(information[:-1], information[1:])):
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
    y = 100
    for i, (info, info2) in enumerate(zip(information[:-1], information[1:])):
        bottom, top, left, right = info['loc']
        width = info['width']
        offset = info['offset']
        mask = info['mask']
        frame = info['frame']
        shape = frame.shape
        if info['skip']:
            canvas[- shape[0]:, x: x + width][mask] = frame[mask]
        x += offset

    x = 0
    y = 100
    num_to_draw = int(len(information) / skip_frame)
    count = 0
    for i, info in enumerate(information):
        bottom, top, left, right = info['loc']
        width = info['width']
        offset = info['offset']

        mask = info['mask']
        frame = info['frame']
        shape = frame.shape
        if not info['skip']:
            #             val = frame[:, :, alpha_dim].astype(np.float)
            #             val *= np.power(0.995, num_to_draw - count)
            #             frame[:, :, alpha_dim] = val.astype(np.uint8)
            canvas[- shape[0]:, x: x + width][mask] = frame[mask]
            count += 1
        x += offset

    if draw_last_frame:
        info = information[-1]
        bottom, top, left, right = info['loc']
        width = info['width']
        offset = info['offset']

        mask = info['mask']
        frame = info['frame']
        shape = frame.shape
        frame[:, :, alpha_dim] = 255
        canvas[- shape[0]:, x: x + width][mask] = frame[mask]
        count += 1

    # if display:
    #     imshow(canvas, False)

    return canvas


def get_reward(orginal_ckpt, default_th=0.6):
    rew_file = orginal_ckpt.replace("CheckPoints", "Rwds")
    if re.search(r"_\d+_", rew_file):
        return None

    if "EarlyStop" in orginal_ckpt:
        rew_file = rew_file.replace("EarlyStopPolicy_Suc",
                                    "EarlyStopPolicy_Suc_rwds")

        if "0.0drop_prob" not in rew_file:
            #             rew_file = rew_file.replace("0.0drop_prob_", "")

            if "threshold" not in rew_file:
                rew_file = rew_file.replace("hidden_",
                                            "hidden_{}threshold_".format(
                                                default_th))

    else:
        rew_file = rew_file.replace("checkpoint", "rwds")
    with open(rew_file, "r") as f:
        data = csv.reader(f)
        last_line = list(data)[-1]

    return eval(last_line[0])


def read_result_from_ckpt_dir(ckpt_dir, hopper_special_process=False):
    ckpt_list = os.listdir(ckpt_dir)
    ppo_result = []
    our_result = []

    for ckpt in ckpt_list:

        if hopper_special_process:
            print("special process hopper, the ckpt is:", ckpt)
            if "10hidden" not in ckpt:
                continue
            if "0.0drop" not in ckpt:
                continue

        ckpt = os.path.join(ckpt_dir, ckpt)
        rew = get_reward(ckpt)
        if rew is not None:
            if "Early" in ckpt:
                our_result.append([ckpt, rew])
            else:
                ppo_result.append([ckpt, rew])
    return ppo_result, our_result


import time


def collect_frame_batch(pair_list, vis_env, agent_name, num_steps=200,
                        verbose=True, threshold=0):
    start = now = time.time()
    oidlist = {}

    ret_dict = {}
    count = 0

    for ckpt, rew in pair_list:
        count += 1
        """
        ckpt looks like: 
        Train_PPO_walker/Hopper/CheckPoints/checkpoint_10hidden_0
        .0drop_prob_0repeat
        the reward files look like: 
            Train_PPO_walker/Hopper/Rwds/EarlyStopPolicy_Suc_rwds_10hidden_0
            .9threshold_16repeat
        """
        if verbose:
            print("[+{:.2f}/{:.2f}s] Current figure: {}, rew {:.2f}.".format(
                time.time() - now, time.time() - start,
                ckpt, rew))
            now = time.time()

        new_frame_list, velocity, extra_info_dict, frames_dict = \
            collect_frame(ckpt, agent_name, vis_env, num_steps=num_steps,
                          threshold=threshold)

        ret_dict[ckpt] = {
            "frame": new_frame_list,
            "velocity": velocity,
            "ckpt": ckpt,
            "frames_dict": frames_dict,
            "reward": rew,
            "extra_info_dict": extra_info_dict,
            "real_reward": extra_info_dict['reward'][agent_name][-1]
        }
        if verbose:
            print(
                "({}/{}) [+{:.2f}/{:.2f}s] Collected {} frames. Reward:"
                "{:.2f}".format(
                    count, len(pair_list),
                    time.time() - now, time.time() - start,
                    len(new_frame_list), rew)
            )
            now = time.time()
    return ret_dict


def collect(ckpt_dir_path, agent_name, vis_env, num_steps=200, num_ckpt=20,
            threshold=0, hopper_special_process=False):

    ppo_result, our_result = read_result_from_ckpt_dir(
        ckpt_dir_path, hopper_special_process
    )

    if hopper_special_process:
        num_ckpt = 10

    pairs = sorted(our_result, key=lambda x: x[1])[-num_ckpt:]
    frame_dict_our_hopper = collect_frame_batch(pairs, vis_env, agent_name,
                                                num_steps=num_steps,
                                                threshold=threshold)

    ppo_result_pairs = sorted(ppo_result, key=lambda x: x[1])[-num_ckpt:]
    frame_dict_our_hopper_ppo = collect_frame_batch(ppo_result_pairs, vis_env,
                                                    agent_name,
                                                    num_steps=num_steps,
                                                    threshold=threshold)

    return frame_dict_our_hopper, frame_dict_our_hopper_ppo


DEFAULT_CONFIG = dict(
    start=0,
    interval=190,
    skip_frame=20,
    alpha=0.48,
    velocity_multiplier=7
)


def draw_all_result(result_dict, choose_index=None, config=DEFAULT_CONFIG,
                    threshold=0):
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
                "with index {}".format(
                    reward, threshold, i))
            continue

        if interval is None:
            interval = len(new_frame_list) - start

        if len(new_frame_list) < start + interval:
            print("The index {} agent has too small rollout length {}"
                  " and can not achieve {} frames requirement.".format(
                i, len(new_frame_list), start + interval)
            )
            continue

        index_ckpt_map[i] = k
        fig = draw_one_exp(
            new_frame_list[start: start + interval],
            velocity[start: start + interval], alpha=alpha,
            clip_top=100, skip_frame=skip_frame,
            velocity_multiplier=velocity_multiplier, display=True,
            draw_last_frame=False, num_inter=0
        )
        fig_dict[k] = fig
        print(
            "(LOOK UP) Current index: {}, Claim Reward: {}, Real Reward:"
            "{}\n\n".format(
                i, v['reward'], v['real_reward']
            ))

    return index_ckpt_map, fig_dict


def plot(index_ckpt_map, fig_dict, result_dict, choose=None, width=5000,
         clip=100, put_text=True):
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
                cv2.FONT_HERSHEY_SIMPLEX, 2.2,
                (0, 0, 0, 255), 2
            )
        else:
            frame_return = frame[clip:, :clip_width].copy()

        canvas[i * height: (i + 1) * height, :clip_width, ...] = frame_return

    return canvas
