import numpy as np
import pandas
import ray
from scipy.fftpack import fft

from rollout import rollout
from utils import restore_agent


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
    def __init__(self, run_name, ckpt, env_name, env_maker, agent_name):
        # restore an agent here
        self.agent = restore_agent(run_name, ckpt, env_name)
        self.env_maker = env_maker
        self.agent_name = agent_name
        self._num_steps = None

    def _get_representation1(self, df):
        # M sequence whose length is NL
        mean = df.groupby(['rollout', 'frequency']).mean().value.to_numpy()
        std = df.groupby(['rollout', 'frequency']).std().value.to_numpy()
        return mean, std

    def _get_representation2(self, df):
        # MN sequence whose length is L
        mean = df.groupby('frequency').mean().value.to_numpy()
        std = df.groupby('frequency').std().value.to_numpy()
        return mean, std

    def _get_representation3(self, df):
        # M sequence which are averaged sequence (length L) across N
        n_averaged = df.groupby(['frequency', 'seed']).mean()
        # now n_averaged looks like:
        """
                            value  rollout
        frequency seed                   
        0         0    -0.850097        2
                  1    -0.812924        2
                  2    -0.810672        2
        1         0    -2.704238        2
                  1    -2.565724        2
                          ...      ...
        48        1    -5.055128        2
                  2    -5.066221        2
        49        0    -4.989105        2
                  1    -5.000202        2
                  2    -5.088004        2
        [150 rows x 2 columns]
        """
        mean = n_averaged.value.mean(level='frequency').to_numpy()
        std = n_averaged.value.std(level='frequency').to_numpy()
        return mean, std

    def _rollout(self, env):
        ret = rollout(
            self.agent,
            env,
            require_trajectory=True,
            num_steps=self._num_steps or 0
        )
        ret = ret["trajectory"]
        obs = np.array([a[0] for a in ret])
        act = np.array([a[1] for a in ret])
        return obs, act

    def _rollout_multiple(
            self, num_rollouts, seed, stack=False, normalize=True, log=True
    ):
        # One seed, N rollouts.
        env = self.env_maker()
        env.seed(seed)
        data_frame = None
        obs_list = []
        act_list = []
        for i in range(num_rollouts):
            print(
                "Agent {}, Rollout {}/{}".format(
                    self.agent_name, i, num_rollouts
                )
            )
            obs, act = self._rollout(env)
            if not stack:
                df = stack_fft(obs, act, normalize=normalize, use_log=log)
                df.insert(df.shape[1], "rollout", i)
                data_frame = df if data_frame is None else \
                    data_frame.append(df, ignore_index=True)
            else:
                obs_list.append(obs)
                act_list.append(act)
        if stack:
            data_frame = stack_fft(
                np.concatenate(obs_list),
                np.concatenate(act_list),
                normalize=normalize,
                use_log=log
            )
            data_frame.insert(data_frame.shape[1], "rollout", 0)
        data_frame.insert(data_frame.shape[1], "agent", self.agent_name)
        data_frame.insert(data_frame.shape[1], "seed", seed)
        return data_frame

    def _get_representation(self, data_frame, stack):
        representation_form = {
            # M sequence whose length is NL
            "M_sequenceNL": self._get_representation1,
            # MN sequence whose length is L
            "MN_sequenceL": self._get_representation2,
            # M sequence which are averaged sequence (length L) across N
            "M_N_sequenceL": self._get_representation3
        }
        if stack:
            rep = representation_form['M_sequenceNL'](data_frame)
            return {"M_sequenceNL": rep}
        else:
            result = {}
            for key in ['MN_sequenceL', "M_N_sequenceL"]:
                rep = representation_form[key](data_frame)
                result[key] = rep
            return result

    @ray.method(num_return_vals=2)
    def fft(
            self,
            num_seeds,
            num_rollouts,
            stack=False,
            clip=None,
            normalize=True,
            log=True,
            fillna=0,
            _num_steps=None
    ):
        if _num_steps:
            # For testing purpose
            self._num_steps = _num_steps
        data_frame = None
        for seed in range(num_seeds):
            df = self._rollout_multiple(
                num_rollouts, seed, stack, normalize, log
            )
            if data_frame is None:
                data_frame = df
            else:
                data_frame = data_frame.append(df, ignore_index=True)
        # print("df shape", data_frame.shape)
        if clip:
            pass
            # TODO
        data_frame.fillna(fillna, inplace=True)
        return data_frame, self._get_representation(data_frame, stack)


if __name__ == '__main__':
    from gym.envs.box2d import BipedalWalker

    ray.init(local_mode=True)

    fft_worker = FFTWorker.remote(
        "PPO",
        "~/ray_results/0810-20seeds/PPO_BipedalWalker-v2_0_seed=0_2019-08"
        "-10_15-21-164grca382/checkpoint_313/checkpoint-313",
        "BipedalWalker-v2", BipedalWalker, "TEST_AGENT"
    )

    oid1, oid2 = fft_worker.fft.remote(3, 5, False)

    df = ray.get(oid1)
    rep = ray.get(oid2)

    # ray.shutdown()
