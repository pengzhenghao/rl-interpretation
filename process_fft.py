import collections

import gym
import numpy as np
import pandas

from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.evaluation.episode import _flatten_action
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from scipy.fftpack import fft

from utils import DefaultMapping

# %matplotlib inline
# import seaborn as sns
# import ray
# ray.init(ignore_reinit_error=True, log_to_driver=)

# def restore(ckpt):
#     config = {}
#     cls = get_agent_class('PPO')
#     agent = cls(env="BipedalWalker-v2", config=config)
#     agent.restore(ckpt)
#     return agent


def default_policy_agent_mapping(unused_agent_id):
    return DEFAULT_POLICY_ID


def rollout(agent, env_name, num_steps, seed=0, out=None, no_render=True):
    # Mostly copied from ray.rllib.rollout
    # But we allow the function quit when one episode is ended.
    policy_agent_mapping = default_policy_agent_mapping

    if hasattr(agent, "workers"):
        env = agent.workers.local_worker().env
        multiagent = isinstance(env, MultiAgentEnv)
        if agent.workers.local_worker().multiagent:
            policy_agent_mapping = agent.config["multiagent"
                                                ]["policy_mapping_fn"]

        policy_map = agent.workers.local_worker().policy_map
        state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
        use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
        action_init = {
            p: m.action_space.sample()
            for p, m in policy_map.items()
        }
    else:
        env = gym.make(env_name)
        multiagent = False
        use_lstm = {DEFAULT_POLICY_ID: False}

    env.seed(seed)
    if out is not None:
        rollouts = []
    steps = 0
    #     while steps < (num_steps or steps + 1):
    mapping_cache = {}  # in case policy_agent_mapping is stochastic
    if out is not None:
        rollout = []
    obs = env.reset()
    agent_states = DefaultMapping(
        lambda agent_id: state_init[mapping_cache[agent_id]]
    )
    prev_actions = DefaultMapping(
        lambda agent_id: action_init[mapping_cache[agent_id]]
    )
    prev_rewards = collections.defaultdict(lambda: 0.)
    done = False
    reward_total = 0.0
    rollout = []
    while not done and steps < (num_steps or steps + 1):
        multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}
        action_dict = {}
        for agent_id, a_obs in multi_obs.items():
            if a_obs is not None:
                policy_id = mapping_cache.setdefault(
                    agent_id, policy_agent_mapping(agent_id)
                )
                p_use_lstm = use_lstm[policy_id]
                if p_use_lstm:
                    a_action, p_state, _ = agent.compute_action(
                        a_obs,
                        state=agent_states[agent_id],
                        prev_action=prev_actions[agent_id],
                        prev_reward=prev_rewards[agent_id],
                        policy_id=policy_id
                    )
                    agent_states[agent_id] = p_state
                else:
                    a_action = agent.compute_action(
                        a_obs,
                        prev_action=prev_actions[agent_id],
                        prev_reward=prev_rewards[agent_id],
                        policy_id=policy_id
                    )
                a_action = _flatten_action(a_action)  # tuple actions
                action_dict[agent_id] = a_action
                prev_actions[agent_id] = a_action
        action = action_dict

        action = action if multiagent else action[_DUMMY_AGENT_ID]
        next_obs, reward, done, _ = env.step(action)
        if multiagent:
            for agent_id, r in reward.items():
                prev_rewards[agent_id] = r
        else:
            prev_rewards[_DUMMY_AGENT_ID] = reward

        if multiagent:
            done = done["__all__"]
            reward_total += sum(reward.values())
        else:
            reward_total += reward
        rollout.append([obs, action, next_obs, reward, done])
        steps += 1
        obs = next_obs
    return rollout


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
    ret = rollout(agent, "BipedalWalker-v2", 0, seed)
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
