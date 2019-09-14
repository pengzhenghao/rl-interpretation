"""
This file is used for conduct inverse fft.
Given the fft representation's cluster centroid,
we can inversely generate the motion of the fake-agent.

The way we do that is first we collect the frequency pattern of
the most-closed agent, then we use the centroid representation to replace
the magnitude pattern of the representative agent. Now we have both magnitude
and phase pattern, we can conduct the inverse fft to get the trial.

By modifying the environment, we can generate the videos from the trial.

The algorithm procedure:

For each cluster:
    1. get the closest agent "a"
    2. collect the frequency pattern of a (by rollout N times)
    3. replace the magnitude pattern by the centroid representation.
    4. split the representation into 24 + 4 channels and generate the sequences
    5. generate the videos based on the regenerated obs / act sequence.


"""

import numpy as np
import pandas
from scipy.fftpack import fft, ifft
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from utils import initialize_ray


def build_env():
    env = BipedalWalker()
    env.seed(0)
    return env


FFT_ENV_MAKER_LOOKUP = {"BipedalWalker-v2": build_env}


def compute_fft(y):
    y = np.asarray(y)
    assert y.ndim == 1
    yy = fft(y)
    mag = np.abs(yy)
    phase = yy / mag
    return mag, phase


def inverse(magnitude, phase):
    return ifft(magnitude * np.exp(1j * phase)).real


def stack_fft(obs, act, normalize, use_log=True):
    obs = np.asarray(obs)
    act = np.asarray(act)

    if normalize:
        if normalize is True:
            normalize = "range"
        assert isinstance(normalize, str)
        assert normalize in ["std", "range"]
        if normalize == "std":
            obs = StandardScaler().fit_transform(obs)
            act = StandardScaler().fit_transform(act)
        elif normalize == "range":
            obs = MinMaxScaler().fit_transform(obs)
            act = MinMaxScaler().fit_transform(act)

    def parse(x):
        if use_log:
            return np.log(x + 1e-12)
        else:
            return x

    result = {}
    data_col = []
    label_col = []
    frequency_col = []

    # obs = [total timesteps, num of channels]
    for ind, y in enumerate(obs.T):
        yf2 = compute_fft(y)
        yf2 = parse(yf2)
        data_col.append(yf2)
        label_col.extend(["Obs {}".format(ind)] * len(yf2))
        frequency_col.append(np.arange(len(yf2)))

    for ind, y in enumerate(act.T):
        yf2 = compute_fft(y)
        yf2 = parse(yf2)
        data_col.append(yf2)
        label_col.extend(["Act {}".format(ind)] * len(yf2))
        frequency_col.append(np.arange(len(yf2)))

    result['frequency'] = np.concatenate(frequency_col)
    result['value'] = np.concatenate(data_col)
    result['label'] = label_col

    return pandas.DataFrame(result)


if __name__ == '__main__':
    from gym.envs.box2d import BipedalWalker
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml-path", type=str, required=True)
    parser.add_argument("--num-rollouts", type=int, default=100)
    parser.add_argument("--num-agents", type=int, default=-1)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--show", action="store_true", default=False)
    args = parser.parse_args()
    initialize_ray(False)

    get_fft_cluster_finder(
        yaml_path=args.yaml_path,
        normalize="std",
        num_agents=None if args.num_agents == -1 else args.num_agents,
        num_rollouts=args.num_rollouts,
        num_workers=args.num_workers,
        show=args.show
    )
