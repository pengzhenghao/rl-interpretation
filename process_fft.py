import copy
import time
from math import ceil

import numpy as np
import pandas
import ray
from scipy.fftpack import fft
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from process_cluster import ClusterFinder
from process_data import get_name_ckpt_mapping
from rollout import rollout
from utils import restore_agent, initialize_ray, get_random_string


def compute_fft(y):
    y = np.asarray(y)
    assert y.ndim == 1
    yy = fft(y)
    yf = np.abs(yy)
    yf1 = yf / len(y)
    yf2 = yf1[:int(len(y) / 2)]
    return yf2


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


def get_representation(data_frame):
    multi_index_series = data_frame.groupby(["label", 'frequency'])
    mean = multi_index_series.mean().value.reset_index()
    std = multi_index_series.std().value.reset_index()

    # mean.groupby("label").get_group(label).value

    return mean.copy(), std.copy()


@ray.remote
class FFTWorker(object):
    def __init__(self):
        self._num_steps = None
        self.agent = None
        self.agent_name = None
        self.env_maker = None
        self.worker_name = "Untitled Worker"
        self.initialized = False

    @ray.method(num_return_vals=0)
    def reset(
            self, run_name, ckpt, env_name, env_maker, agent_name, worker_name
    ):
        self.initialized = True
        self.agent = restore_agent(run_name, ckpt, env_name)
        self.env_maker = env_maker
        self.agent_name = agent_name
        self._num_steps = None
        self.worker_name = worker_name
        print("{} is reset!".format(worker_name))

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
            self,
            num_rollouts,
            seed,
            stack=False,
            normalize="range",
            log=True,
            _num_seeds=None,
            _extra_name=""
    ):
        # One seed, N rollouts.
        env = self.env_maker()
        env.seed(seed)
        data_frame = None
        obs_list = []
        act_list = []
        for i in range(num_rollouts):
            print(
                "({}) Agent {}<{}>, Seed {}, Rollout {}/{}".format(
                    self.worker_name, _extra_name, self.agent_name,
                    seed if _num_seeds is None else
                    "No.{}/{} (Real: {})".format(seed + 1, _num_seeds, seed),
                    i, num_rollouts
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

    @ray.method(num_return_vals=2)
    def fft(
            self,
            num_seeds,
            num_rollouts,
            stack=False,
            clip=None,
            normalize="range",
            log=True,
            fillna=0,
            _num_steps=None,
            _extra_name=""
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
        :param _extra_name:
        :return:
        The representation form:

        {
            method1: DataFrame(
                    col=[label, agent, value, frequency, seed]
                ),
            method2: DataFrame()
        }

        """
        assert self.initialized, "You should reset the worker first!"
        if _num_steps:
            # For testing purpose
            self._num_steps = _num_steps
        if normalize:
            assert isinstance(normalize, str)
            assert normalize in ['range', 'std']
        data_frame = None
        for seed in range(num_seeds):
            df = self._rollout_multiple(
                num_rollouts,
                seed,
                stack,
                normalize,
                log,
                _num_seeds=num_seeds,
                _extra_name=_extra_name
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
        return data_frame.copy(), get_representation(data_frame)


def get_fft_representation(
        name_ckpt_mapping,
        run_name,
        env_name,
        env_maker,
        num_seeds,
        num_rollouts,
        stack=False,
        normalize="range",
        num_worker=10
):
    initialize_ray()

    data_frame_dict = {}
    representation_dict = {}

    num_agents = len(name_ckpt_mapping)

    num_iteration = int(ceil(num_agents / num_worker))

    agent_ckpt_dict_range = list(name_ckpt_mapping.items())
    agent_count = 1
    agent_count_get = 1

    workers = [FFTWorker.remote() for _ in range(num_worker)]
    now_t_get = now_t = start_t = time.time()

    for iteration in range(num_iteration):
        start = iteration * num_worker
        end = min((iteration + 1) * num_worker, num_agents)
        df_obj_ids = []
        rep_obj_ids = []
        for i, (name, ckpt) in enumerate(agent_ckpt_dict_range[start:end]):
            workers[i].reset.remote(
                run_name=run_name,
                ckpt=ckpt,
                env_name=env_name,
                env_maker=env_maker,
                agent_name=name,
                worker_name="Worker{}".format(i)
            )

            df_obj_id, rep_obj_id = workers[i].fft.remote(
                num_seeds,
                num_rollouts,
                stack,
                normalize=normalize,
                _extra_name="[{}/{}] ".format(agent_count, num_agents)
            )
            print(
                "[{}/{}] (+{:.1f}s/{:.1f}s) Start collecting data from agent <{}>".
                format(
                    agent_count_get, num_agents,
                    time.time() - now_t,
                    time.time() - start_t, name
                )
            )
            agent_count += 1
            now_t = time.time()

            df_obj_ids.append(df_obj_id)
            rep_obj_ids.append(rep_obj_id)
        for df_obj_id, rep_obj_id, (name, _) in zip(
                df_obj_ids, rep_obj_ids, agent_ckpt_dict_range[start:end]):
            df = ray.get(df_obj_id)
            rep = ray.get(rep_obj_id)
            data_frame_dict[name] = copy.deepcopy(df)
            representation_dict[name] = copy.deepcopy(rep)
            del df
            del rep
            print(
                "[{}/{}] (+{:.1f}s/{:.1f}s) Got data from agent <{}>".format(
                    agent_count_get, num_agents,
                    time.time() - now_t_get,
                    time.time() - start, name
                )
            )
            agent_count_get += 1
            now_t_get = time.time()
    return data_frame_dict, representation_dict


def parse_representation_dict(
        representation_dict,
        padding="fix",
        padding_length=None,
        padding_value=None
):
    data_frame = None
    for agent_name, rep_dict in representation_dict.items():
        df = rep_dict[0]
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

    if padding == 'fix':
        assert padding_length is not None
        assert padding_value is not None

        def pad(vec):
            vec = np.asarray(vec)
            assert vec.ndim == 1
            vec[np.isnan(vec)] = padding_value
            back = np.empty((padding_length, ))
            back.fill(padding_value)
            end = min(len(vec), padding_length)
            back[:end] = vec[:end]
            return back
    else:
        raise NotImplementedError("Only support fix padding now.")

    for agent in agent_list:
        arr_list = []
        for label in label_list:
            mask = (data_frame.agent == agent) & (data_frame.label == label)
            array = data_frame.value[mask].to_numpy()
            array = pad(array)
            arr_list.append(array)
        filled_dict[agent] = arr_list
        filled_flat_dict[agent] = np.concatenate(arr_list)
        print("Finished parse data of agent <{}>".format(agent))



    cluster_df = pandas.DataFrame.from_dict(filled_flat_dict).T
    # cluster_df means a matrix each row is an agent representation.
    return cluster_df.copy()


def get_fft_cluster_finder(
        yaml_path,
        env_name,
        env_maker,
        run_name,
        normalize,
        num_agents=None,
        num_seeds=1,
        num_rollouts=100,
        padding="fix",
        padding_length=500,
        padding_value=0,
        show=False
):
    assert yaml_path.endswith('yaml')

    name_ckpt_mapping = get_name_ckpt_mapping(yaml_path, num_agents)
    print("Successfully loaded name_ckpt_mapping!")

    num_agents = num_agents or len(name_ckpt_mapping)
    # prefix: data/XXX_10agent_100rollout_1seed_28sm29sk
    prefix = "".join(
        [
            yaml_path.split('.yaml')[0],
            "_{}agents_{}rollout_{}seed_{}".format(
                num_agents, num_rollouts, num_seeds, get_random_string()
            )
        ]
    )

    data_frame_dict, repr_dict = get_fft_representation(
        name_ckpt_mapping,
        run_name,
        env_name,
        env_maker,
        num_seeds,
        num_rollouts,
        normalize=normalize
    )
    print("Successfully get FFT representation!")

    cluster_df = parse_representation_dict(repr_dict, padding, padding_length, padding_value)
    print("Successfully get cluster dataframe!")

    # Store
    assert isinstance(cluster_df, pandas.DataFrame)
    pkl_path = prefix + '.pkl'
    cluster_df.to_pickle(pkl_path)
    print("Successfully store cluster_df! Save at: {}".format(pkl_path))

    # Cluster
    nostd_cluster_finder = ClusterFinder(cluster_df, standardize=False)
    nostd_fig_path = prefix + '_nostd.png'
    nostd_cluster_finder.display(save=nostd_fig_path, show=show)
    print(
        "Successfully finish no-standardized clustering! Save at: {}".
        format(nostd_fig_path)
    )

    std_cluster_finder = ClusterFinder(cluster_df, standardize=True)
    std_fig_path = prefix + "_std.png"
    std_cluster_finder.display(save=std_fig_path, show=show)
    print(
        "Successfully finish standardized clustering! Save at: {}".
        format(std_fig_path)
    )

    ret = {
        "cluster_finder": {
        'nostd_cluster_finder': nostd_cluster_finder,
        "std_cluster_finder": std_cluster_finder
    },
        "prefix": prefix
    }

    return ret


if __name__ == '__main__':
    from gym.envs.box2d import BipedalWalker

    initialize_ray(True)

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
