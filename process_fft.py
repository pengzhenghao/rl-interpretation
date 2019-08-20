import numpy as np
import pandas

from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from scipy.fftpack import fft
import gym
from rollout import rollout
import ray

def compute_fft(y, normalize=True, normalize_max=None, normalize_min=None):
    y = np.asarray(y)
    assert y.ndim == 1

    if normalize:
        if normalize_min is None:
            normalize_min = y.min()
        if normalize_max is None:
            normalize_max = y.max()
        y = (y - normalize_min) / (normalize_max - normalize_min)

    yy = fft(y)
    yf = np.abs(yy)
    yf1 = yf / len(y)
    yf2 = yf1[:int(len(y) / 2)]
    return yf2

def stack_fft(obs, act, normalize, use_log=True):
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

@ray.remote
class FFTWorker(object):

    def __init__(self, ckpt):
        # restore an agent here
        self.agent = None
        self.env_maker = None
        self.agent_name = None
        pass

    def _rollout(self, env):
        ret = rollout(self.agent, env, require_trajectory=True)
        ret = ret["trajectory"]
        obs = np.array([a[0] for a in ret])
        act = np.array([a[1] for a in ret])
        return obs, act

    def _rollout_multiple(self, number_of_rollouts, seed,
                          stack=False, normalize=True):
        # One seed, N rollouts.
        env = self.env_maker()
        env.seed(seed)
        data_frame = None
        obs_list = []
        act_list = []
        for i in range(number_of_rollouts):
            print("Agent {}, Rollout {}/{}".format(self.agent_name, i, number_of_rollouts))
            obs, act = self._rollout(env)
            if not stack:
                df = stack_fft(obs, act, normalize=normalize)
                df.insert(df.shape[1], "seed", seed)
                data_frame = df if data_frame is None else \
                    data_frame.append(df, ignore_index=True)
            else:
                obs_list.append(obs)
                act_list.append(act)
        if stack:
            data_frame = stack_fft(
                np.concatenate(obs_list),
                np.concatenate(act_list),
                normalize=normalize
            )
        data_frame.insert(data_frame.shape[1], "agent", self.agent_name)
        return data_frame

    def _get_representation(self, data_frame):
        representation = None
        # TODO
        return representation

    @ray.method(num_return_vals=1)
    def fft(self, num_seeds, num_rollouts, stack=False, clip=None, normalize=True,
            log=True):
        data_frame = None
        for seed in range(num_seeds):
            df = self._rollout_multiple(num_rollouts, seed, stack, normalize)
            if data_frame is None:
                data_frame = df
            else:
                data_frame = data_frame.append(df, ignore_index=True)
        if clip:
            pass
            # TODO
        return self._get_representation(data_frame)
