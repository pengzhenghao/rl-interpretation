import numpy as np
import pandas

from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from scipy.fftpack import fft
import gym
from rollout import rollout


def default_policy_agent_mapping(unused_agent_id):
    return DEFAULT_POLICY_ID


def compute_fft(y, normalize=True):
    y = np.asarray(y)
    assert y.ndim == 1

    if normalize:
        y = (y - y.min()) / (y.max() - y.min())

    yy = fft(y)  # 快速傅里叶变换

    yf = np.abs(yy)  # 取绝对值
    yf1 = yf / len(y)  # 归一化处理
    yf2 = yf1[:int(len(y) / 2)]  # 由于对称性，只取一半区间
    return yf2


def stack_fft(obs, act, use_log=True, normalize=True):
    obs = np.asarray(obs)
    act = np.asarray(act)

    parse = lambda x: np.log(x + 1e-12) if use_log else lambda x: x

    result = {}

    data_col = []
    label_col = []
    frequency_col = []

    for ind, y in enumerate(obs.T):
        yf2 = compute_fft(y, normalize)
        yf2 = parse(yf2)

        data_col.append(yf2)
        label_col.extend(["Obs {}".format(ind)] * len(yf2))
        frequency_col.append(np.arange(len(yf2)))

    for ind, y in enumerate(act.T):
        yf2 = compute_fft(y, normalize)
        yf2 = parse(yf2)

        data_col.append(yf2)
        label_col.extend(["Act {}".format(ind)] * len(yf2))
        frequency_col.append(np.arange(len(yf2)))

    result['frequency'] = np.concatenate(frequency_col)
    result['value'] = np.concatenate(data_col)
    result['label'] = label_col

    return pandas.DataFrame(result)


def rollout_once(agent, seed=0):
    env = gym.make("BipedalWalker-v2")
    env.seed(seed)
    ret = rollout(agent, env, require_trajectory=True)
    ret = ret["trajectory"]
    obs = np.array([a[0] for a in ret])
    act = np.array([a[1] for a in ret])
    return obs, act


def rollout_multiple(
        agent_name, agent, seed_list, normalize=True, stack_rollout=True
):
    assert isinstance(seed_list, list)
    data_frame = None
    obs_list = []
    act_list = []
    for seed in seed_list:
        print("Current Testing Agent: <{}>, Seed: {}".format(agent_name, seed))
        obs, act = rollout_once(agent, seed)
        if not stack_rollout:
            df = stack_fft(obs, act, normalize=normalize)
            df.insert(df.shape[1], "seed", seed)
            data_frame = df if data_frame is None else \
                data_frame.append(df, ignore_index=True)
        else:
            obs_list.append(obs)
            act_list.append(act)
    if stack_rollout:
        data_frame = stack_fft(
            np.concatenate(obs_list),
            np.concatenate(act_list),
            normalize=normalize
        )
    data_frame.insert(data_frame.shape[1], "agent", agent_name)
    return data_frame


def evaluate_agents_frequency_character(
        agent_dict, seed_list, normalize=True, stack_rollout=False
):
    data_frame = None
    for agent_name, agent in agent_dict.items():
        df = rollout_multiple(
            agent_name, agent, seed_list, normalize, stack_rollout
        )
        if data_frame is None:
            data_frame = df
        else:
            data_frame = data_frame.append(df, ignore_index=True)
    return data_frame


evaluate_different_agents = evaluate_agents_frequency_character
