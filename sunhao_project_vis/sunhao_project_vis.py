import csv
import math
import os
import re
import time
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from toolbox.env.env_maker import get_env_maker
from toolbox.visualize.multiple_exposure import draw_one_exp, collect_frame

SUNHAO_AGENT_NAME = "SunAgentWrapper"


class conf(object):
    def __init__(self, hid_num=30, drop_prob=0.1):
        self.hid_num = hid_num
        self.drop_prob = drop_prob


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, config, layer_norm=True):
        super(ActorCritic, self).__init__()

        self.actor_fc1 = nn.Linear(num_inputs, 32)
        self.actor_fc2 = nn.Linear(32, config.hid_num)
        self.actor_fc3 = nn.Linear(config.hid_num, num_outputs)
        self.actor_logstd = nn.Parameter(torch.zeros(1, num_outputs))

        self.critic_fc1 = nn.Linear(num_inputs, 64)
        self.critic_fc2 = nn.Linear(64, 64)
        self.critic_fc3 = nn.Linear(64, 1)

        if layer_norm:
            self.layer_norm(self.actor_fc1, std=1.0)
            self.layer_norm(self.actor_fc2, std=1.0)
            self.layer_norm(self.actor_fc3, std=0.01)

            self.layer_norm(self.critic_fc1, std=1.0)
            self.layer_norm(self.critic_fc2, std=1.0)
            self.layer_norm(self.critic_fc3, std=1.0)

        self.config = config

    @staticmethod
    def layer_norm(layer, std=1.0, bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)

    def forward(self, states):
        """
        run policy network (actor) as well as value network (critic)
        :param states: a Tensor2 represents states
        :return: 3 Tensor2
        """
        action_mean, action_logstd = self._forward_actor(states)
        critic_value = self._forward_critic(states)
        return action_mean, action_logstd, critic_value

    def _forward_actor(self, states):
        x = torch.tanh(self.actor_fc1(states))
        x = torch.tanh(self.actor_fc2(x))
        x = F.dropout(x, p=self.config.drop_prob, training=self.training)
        action_mean = torch.tanh(self.actor_fc3(x))
        action_logstd = self.actor_logstd.expand_as(action_mean)
        return action_mean, action_logstd

    def _forward_critic(self, states):
        x = torch.tanh(self.critic_fc1(states))
        x = torch.tanh(self.critic_fc2(x))
        critic_value = self.critic_fc3(x)
        return critic_value

    def select_action(self, action_mean, action_logstd, return_logproba=True):
        """
        given mean and std, sample an action from normal(mean, std)
        also returns probability of the given chosen
        """
        action_std = torch.exp(action_logstd)
        action = torch.normal(action_mean, action_std)
        if return_logproba:
            logproba = self._normal_logproba(action, action_mean,
                                             action_logstd, action_std)
        return action, logproba

    @staticmethod
    def _normal_logproba(x, mean, logstd, std=None):
        if std is None:
            std = torch.exp(logstd)

        std_sq = std.pow(2)
        logproba = - 0.5 * math.log(2 * math.pi) - logstd - (x - mean).pow(
            2) / (2 * std_sq)
        return logproba.sum(1)

    def get_logproba(self, states, actions):
        """
        return probability of chosen the given actions under corresponding
        states of current network
        :param states: Tensor
        :param actions: Tensor
        """
        action_mean, action_logstd = self._forward_actor(states)
        action_mean = action_mean.cpu()
        action_logstd = action_logstd.cpu()
        logproba = self._normal_logproba(actions, action_mean, action_logstd)
        return logproba


class args(object):
    #     env_name = ENV_NAME
    seed = 1234
    num_episode = 800
    batch_size = 2048
    max_step_per_round = 2000
    gamma = 0.995
    lamda = 0.97
    log_num_episode = 1
    num_epoch = 10
    minibatch_size = 256
    clip = 0.2
    loss_coeff_value = 0.5
    loss_coeff_entropy = 0.01
    lr = 3e-4
    num_parallel_run = 1
    # tricks
    schedule_adam = 'linear'
    schedule_clip = 'linear'
    layer_norm = True
    state_norm = False
    advantage_norm = True
    lossvalue_norm = True


class FakeActionSpace(object):
    def sample(self):
        return None


class SunPolicyWrapper(object):
    def __init__(self):
        self.action_space = FakeActionSpace()


class SunAgentWrapper(object):
    _name = SUNHAO_AGENT_NAME

    def __init__(self, ckpt, env_name):
        env = get_env_maker(env_name)()
        num_inputs = env.observation_space.shape[0]
        num_actions = env.action_space.shape[0]
        num_units = eval(re.search(r"_(\d+)hidden", ckpt).group(1))

        self.network = ActorCritic(num_inputs, num_actions,
                                   config=conf(num_units, 0.0),
                                   layer_norm=args.layer_norm)
        self.restore(ckpt)
        self.config = {'env': env_name}
        self.policy = SunPolicyWrapper()

    def compute_action(self, a_obs, prev_action=None, prev_reward=None,
                       policy_id=None, full_fetch=None):
        self.network.eval()
        action_mean, action_logstd, value = self.network((
            torch.as_tensor(
                a_obs,
                dtype=torch.float32)).unsqueeze(
            0)
        )
        action = action_mean.data.numpy()[0]
        a_info = {
            "vf_preds": 0,
        }
        return action, None, a_info

    def stop(self):
        pass

    def restore(self, ckpt):
        self.network.load_state_dict(torch.load(ckpt))


def read_ckpt_dir(ckpt_dir, env_name):
    hopper_special_process = env_name.startswith("Hopper")

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

        threshold = 0.6 if env_name.startswith("Hopper") else (
            1.3 if env_name.startswith("HalfCheetah") else 1.1  # Walker
        )

        rew = _read_reward_file(ckpt, threshold)
        if rew is not None:
            if "Early" in ckpt:
                our_result.append([ckpt, rew])
            else:
                ppo_result.append([ckpt, rew])
    return ppo_result, our_result


def _read_reward_file(orginal_ckpt, default_th):
    rew_file = orginal_ckpt.replace("CheckPoints", "Rwds")
    if re.search(r"_\d+_", rew_file):
        return None

    if "EarlyStop" in orginal_ckpt:
        rew_file = rew_file.replace("EarlyStopPolicy_Suc",
                                    "EarlyStopPolicy_Suc_rwds")

        if "0.0drop_prob" not in rew_file:
            if ("reward_threshold" not in rew_file) and (
                    "threshold" not in rew_file):
                rew_file = rew_file.replace("hidden_",
                                            "hidden_{}threshold_".format(
                                                default_th))
    else:
        rew_file = rew_file.replace("checkpoint", "rwds")
    with open(rew_file, "r") as f:
        data = csv.reader(f)
        last_line = list(data)[-1]
    return eval(last_line[0])


def collect_frame_batch(pair_list, vis_env, agent_name, num_steps=200,
                        verbose=True, threshold=0):
    start = now = time.time()
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
            "reward": rew,
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


def collect_frames_from_ckpt_dir(
        ckpt_dir_path, agent_name, vis_env,
        num_steps=200, num_ckpt=20,
        reward_threshold=0,
        hopper_special_process=False
):
    ppo_result, our_result = read_ckpt_dir(
        ckpt_dir_path, vis_env
    )

    if hopper_special_process:
        num_ckpt = 10
    pairs = sorted(our_result, key=lambda x: x[1])[-num_ckpt:]
    frame_dict_our_hopper = collect_frame_batch(pairs, vis_env, agent_name,
                                                num_steps=num_steps,
                                                threshold=reward_threshold)

    ppo_result_pairs = sorted(ppo_result, key=lambda x: x[1])[-num_ckpt:]
    frame_dict_our_hopper_ppo = collect_frame_batch(ppo_result_pairs, vis_env,
                                                    agent_name,
                                                    num_steps=num_steps,
                                                    threshold=reward_threshold)

    return frame_dict_our_hopper, frame_dict_our_hopper_ppo


def draw_all_result(result_dict, choose_index=None, config=None,
                    reward_threshold=0, resize=False):
    index_ckpt_map = {}
    fig_dict = {}
    for i, (k, v) in enumerate(result_dict.items()):
        print("Current Drawing Index {}, Name {}".format(i, k))

        if (choose_index is not None) and (i not in choose_index):
            print(
                "Skip a agent because it's index {} is not in choosen "
                "indices {}."
                    .format(i, choose_index))
            continue
        # if len(v['frame']) > 800 and i > 10:
        #     continue
        new_frame_list = v['frame']
        velocity = v['velocity']

        reward = v['real_reward']
        if reward < reward_threshold:
            print(
                "Reward is {} less than the reward_threshold {}. So we skip "
                "agent "
                "with index {}".format(reward, reward_threshold, i)
            )
            continue

        index_ckpt_map[i] = k
        fig = draw_one_exp(
            new_frame_list,
            velocity,
            config
        )

        if resize:
            size = (fig.shape[1], 300)
            fig = cv2.resize(fig, size, interpolation=cv2.INTER_AREA)

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

    alpha = np.tile(canvas[..., 3:], [1, 1, 3]) / 255  # in range [0, 1]
    white = np.ones_like(alpha) * 255
    return_canvas = np.multiply(canvas[..., :3], alpha).astype(np.uint8) + \
                    np.multiply(white, 1 - alpha).astype(np.uint8)
    return_canvas = cv2.cvtColor(return_canvas, cv2.COLOR_BGR2RGB)
    return return_canvas


def test_sunhao_project_vis():
    """This is the example codes for using our visualization tool at Jupyter
    Notebook.

    The process has three steps:
        1. Load checkpoint, run environment and collect a sequence of frames.
        2. For each agent, generate the multiple-exposure figure.
        3. Concatenate the figure of different agents and add extra
            information like the reward or the agent name (not supported yet).

    It should be note that this codes is organized in the very naive way and
    along with the future development of our toolbox this code may not
    compatible anymore --- That's the reason we left the codes in this branch。

    2019.09.26 Peng Zhenghao
    """

    # Step 1: Load checkpoint and collect frames.
    agent_name = "test-sunhao-0924-halfcheetah"
    vis_env = "HalfCheetah-v3"
    ckpt_dir_path = "Train_PPO_walker/HalfCheetah/CheckPoints"
    halfcheetah_result, halfcheetah_result_ppo = collect_frames_from_ckpt_dir(
        ckpt_dir_path, agent_name, vis_env, num_ckpt=1,
        num_steps=500, reward_threshold=600)

    # Step 2: Draw a multiple-exposure figure based on the frames collect.
    # In this example, we only draw the "our method" result.
    halfcheetah_config = dict(
        start=0,
        interval=300,
        skip_frame=20,
        # alpha=0.48,
        alpha=0.10,
        velocity_multiplier=7
    )
    ret, fig_dict = draw_all_result(halfcheetah_result,
                                    config=halfcheetah_config)

    # Step 3: Concatenate all figures and spread them in a big image.
    canvas = draw_multiple_rows(
        ret,
        fig_dict,
        halfcheetah_result,
        choose=None,
        width=5000,
        clip=100,
        put_text=True
    )

    return canvas


if __name__ == '__main__':
    canvas = test_sunhao_project_vis()
