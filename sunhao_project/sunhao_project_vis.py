import csv
import os
import re
import time
from collections import OrderedDict

from .sunhao_agent_wrapper import SunAgentWrapper

import cv2
import numpy as np

from toolbox.visualize.multiple_exposure import draw_one_exp, collect_frame


def read_ckpt_dir(ckpt_dir, hopper_special_process=False):
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
        rew = _read_reward_file(ckpt)
        if rew is not None:
            if "Early" in ckpt:
                our_result.append([ckpt, rew])
            else:
                ppo_result.append([ckpt, rew])
    return ppo_result, our_result


def _read_reward_file(orginal_ckpt, default_th=0.6):
    rew_file = orginal_ckpt.replace("CheckPoints", "Rwds")
    if re.search(r"_\d+_", rew_file):
        return None

    if "EarlyStop" in orginal_ckpt:
        rew_file = rew_file.replace("EarlyStopPolicy_Suc",
                                    "EarlyStopPolicy_Suc_rwds")

        if "0.0drop_prob" not in rew_file:
            #             rew_file = rew_file.replace("0.0drop_prob_", "")

            if "reward_threshold" not in rew_file:
                rew_file = rew_file.replace("hidden_",
                                            "hidden_{}threshold_".format(
                                                default_th))

    else:
        rew_file = rew_file.replace("checkpoint", "rwds")
    with open(rew_file, "r") as f:
        data = csv.reader(f)
        last_line = list(data)[-1]

    return eval(last_line[0])


def collect(ckpt_dir_path, agent_name, vis_env, num_steps=200, num_ckpt=20,
            threshold=0, hopper_special_process=False):
    ppo_result, our_result = read_ckpt_dir(
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

        agent = SunAgentWrapper(ckpt, vis_env)
        agent.config['env'] = vis_env

        new_frame_list, velocity, extra_info_dict, frames_dict = \
            collect_frame(agent, agent_name, vis_env, num_steps=num_steps,
                          reward_threshold=threshold)

        ret_dict[ckpt] = {
            "frame": new_frame_list,
            "velocity": velocity,
            # "ckpt": ckpt,
            # "frames_dict": frames_dict,
            "reward": rew,
            # "extra_info_dict": extra_info_dict,
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


def draw_all_result(result_dict, choose_index=None, config=None, threshold=0):
    index_ckpt_map = {}
    fig_dict = {}
    for i, (k, v) in enumerate(result_dict.items()):
        if (choose_index is not None) and (i not in choose_index):
            continue
        if len(v['frame']) > 800 and i > 10:
            continue
        new_frame_list = v['frame']
        velocity = v['velocity']

        reward = v['real_reward']
        if reward < threshold:
            print(
                "Reward is {} less than the reward_threshold {}. So we skip "
                "agent "
                "with index {}".format(reward, threshold, i)
            )
            continue

        index_ckpt_map[i] = k
        fig = draw_one_exp(
            new_frame_list,
            velocity,
            config
        )
        fig_dict[k] = fig
        print(
            "(LOOK UP) Current index: {}, Claim Reward: {}, Real Reward:"
            "{}\n\n".format(i, v['reward'], v['real_reward'])
        )

    return index_ckpt_map, fig_dict


def draw_multiple_rows(
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


def test_sunhao():


    index_ckpt_map, fig_dict = \
        draw_all_result(result_dict, choose_index=None, config=None,
                        threshold=0)

    canvas = draw_multiple_rows(
        index_ckpt_map,
        fig_dict,
        result_dict,
        choose=None,
        width=5000,
        clip=100,
        put_text=True
    )

    return canvas


if __name__ == '__main__':
    test_sunhao()
