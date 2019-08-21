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
        multi_index_series = df.groupby(["label", 'rollout', 'frequency'])
        mean = multi_index_series.mean().value.reset_index()
        std = multi_index_series.std().value.reset_index()
        return mean, std

    def _get_representation2(self, df):
        # MN sequence whose length is L
        multi_index_series = df.groupby(["label", 'frequency'])
        mean = multi_index_series.mean().value.reset_index()
        std = multi_index_series.std().value.reset_index()
        return mean, std

    def _get_representation3(self, df):
        # M sequence which are averaged sequence (length L) across N
        n_averaged = df.groupby(['label', 'frequency', 'seed']).mean().value
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
        mean = n_averaged.mean(level=['label', 'frequency']).reset_index()
        std = n_averaged.std(level=['label', 'frequency']).reset_index()
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

        # label_list = data_frame.label.unique()
        # result_dict = {}
        # for label in label_list:
        #     df = data_frame[data_frame.label==label]
        if stack:
            rep = representation_form['M_sequenceNL'](data_frame)
            return {"M_sequenceNL": rep}
        else:
            result = {}
            for key in ['MN_sequenceL', "M_N_sequenceL"]:
                rep = representation_form[key](data_frame)
                result[key] = rep
            return result
        # return result_dict

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
        """

        :param num_seeds:
        :param num_rollouts:
        :param stack:
        :param clip:
        :param normalize:
        :param log:
        :param fillna:
        :param _num_steps:
        :return:
        The representation form:

        {
            method1: DataFrame(
                    col=[label, agent, value, frequency, seed]
                ),
            method2: DataFrame()
        }

        """
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

def parse_MN_sequenceL_result(representation_dict):
    method_name = "MN_sequenceL"
    data_frame = None
    for agent_name, rep_dict in representation_dict.items():
        df = rep_dict[method_name][0]
        df['agent'] = agent_name
        if data_frame is None:
            data_frame = df
        else:
            data_frame = data_frame.append(df, ignore_index=True)

    agent_list = data_frame.agent.unique()
    label_list = data_frame.label.unique()

    label_list = sorted(
        label_list,
        key=lambda s: int(s[4:]) + (-1e6 if s.startswith('Obs') else +1e6)
    )
    # ['Obs 0', 'Obs 1', 'Obs 2', 'Obs 3', 'Obs 4', 'Obs 5', 'Obs 6',
    # 'Obs 7', 'Obs 8', 'Obs 9', 'Obs 10', 'Obs 11', 'Obs 12', 'Obs 13',
    # 'Obs 14', 'Obs 15', 'Obs 16', 'Obs 17', 'Obs 18', 'Obs 19', 'Obs 20',
    # 'Obs 21', 'Obs 22', 'Obs 23', 'Act 0', 'Act 1', 'Act 2', 'Act 3']

    filled_dict = {}
    filled_flat_dict = {}

    def pad(vec, length, val=0):
        vec = np.asarray(vec)
        assert vec.ndim == 1
        vec[np.isnan(vec)] = val
        back = np.empty((length,))
        back.fill(val)
        end = min(len(vec), length)
        back[:end] = vec[:end]
        return back

    for agent in agent_list:
        arr_list = []
        for label in label_list:
            mask = (data_frame.agent == agent) & (data_frame.label == label)
            array = data_frame.value[mask].to_numpy()
            array = pad(array, 500)
            arr_list.append(array)
        filled_dict[agent] = arr_list
        filled_flat_dict[agent] = np.concatenate(arr_list)

    cluster_df = pandas.DataFrame.from_dict(filled_flat_dict).T
    return cluster_df

if __name__ == '__main__':
    from gym.envs.box2d import BipedalWalker

    ray.init(local_mode=True)

    fft_worker = FFTWorker.remote(
        "PPO",
        "~/ray_results/0810-20seeds/PPO_BipedalWalker-v2_0_seed=0_2019-08"
        "-10_15-21-164grca382/checkpoint_313/checkpoint-313",
        "BipedalWalker-v2", BipedalWalker, "TEST_AGENT"
    )

    oid1, oid2 = fft_worker.fft.remote(2, 2, False)

    df = ray.get(oid1)
    rep = ray.get(oid2)

    # ray.shutdown()
