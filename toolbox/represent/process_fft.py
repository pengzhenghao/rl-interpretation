import copy
import os
import os.path as osp
import time
from math import ceil

import numpy as np
import pandas
import ray
from scipy.fftpack import fft
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from toolbox.cluster.process_cluster import ClusterFinder
from toolbox.process_data.process_data import get_name_ckpt_mapping
from toolbox.evaluate.rollout import efficient_rollout_from_worker, make_worker
from toolbox.utils import initialize_ray, get_random_string, ENV_MAKER_LOOKUP
from toolbox.evaluate.evaluate_utils import restore_agent


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


def get_representation(data_frame, label_list, postprocess):
    multi_index_series = data_frame.groupby(["label", 'frequency'])
    mean = multi_index_series.mean().value
    groupby = mean.groupby("label")
    channel_reprs = []
    for label in label_list:
        channel_reprs.append(postprocess(groupby.get_group(label).to_numpy()))
    ret = np.concatenate(channel_reprs)
    return ret


def get_period(source, fps):
    # Compute the period of BipedalWalker-v2's agent.
    # We observe the observation[7, 12] to get the frequency.
    # But this operation is done when rolling out, in order to save memory.
    ret = []
    assert isinstance(source, np.ndarray)
    assert source.ndim == 2
    for i in range(source.shape[1]):
        y = source[:, i]
        fre = compute_fft(y)
        fre[0] = -np.inf
        period = len(y) / fre.argmax()  # * (fps / len(y))  # in Hz
        # period = freq
        ret.append(period)
    return float(np.mean(ret))


set().difference()


@ray.remote(num_gpus=0.3)
class FFTWorker(object):
    def __init__(self):
        self._num_steps = None
        # self.agent = None
        self.agent_name = None
        self.run_name = None
        self.env_name = None
        self.ckpt = None
        self.env_maker = None
        self.worker_name = "Untitled Worker"
        self.initialized = False
        self.rollout_worker = None
        self.postprocess_func = lambda x: x
        self.padding_value = None
        self.seed = 0

    @ray.method(num_return_vals=0)
    def reset(
            self,
            run_name,
            ckpt,
            env_name,
            env_maker,
            agent_name,
            num_rollouts,
            padding=None,
            padding_length=None,
            padding_value=None,
            worker_name=None,
            seed=0,
    ):
        self.initialized = True
        self.run_name = run_name
        self.num_rollouts = num_rollouts
        self.ckpt = ckpt
        self.env_name = env_name
        self.env_maker = env_maker
        self.agent_name = agent_name
        self.seed = seed
        self._num_steps = None
        self.worker_name = worker_name or "Untitled Worker"

        if padding is not None:
            assert padding_value is not None
            assert padding_length is not None

            def pad(vec):
                vec = np.asarray(vec)
                assert vec.ndim == 1
                vec[np.isnan(vec)] = padding_value
                back = np.empty((padding_length, ))
                back.fill(padding_value)
                end = min(len(vec), padding_length)
                back[:end] = vec[:end]
                return back

            self.postprocess_func = pad
        else:
            self.postprocess_func = lambda x: x

        self.padding_value = padding_value
        print("{} is reset!".format(worker_name))

    def _make_rollout_worker(self):
        assert self.initialized

        if self.rollout_worker is None:
            self.rollout_worker = \
                make_worker(self.env_maker, self.ckpt, self.num_rollouts, self.seed,
                            self.run_name, self.env_name)
        else:
            self.rollout_worker.reset(
                self.ckpt, self.num_rollouts, self.seed, self.env_maker,
                self.run_name, self.env_name
            )

    def _efficient_rollout(
            self,
            seed,
    ):
        rollout_result = efficient_rollout_from_worker(
            self.rollout_worker, self.num_rollouts
        )

        print(
            "[FFTWorker._efficient_rollout] Successfully collect "
            "{} rollouts.".format(self.num_rollouts)
        )

        data_frame = None

        for i, roll in enumerate(rollout_result):
            obs, act = roll[0], roll[1]
            df = stack_fft(obs, act, normalize="std", use_log=True)
            df.insert(df.shape[1], "rollout", i)
            data_frame = df if data_frame is None else \
                data_frame.append(df, ignore_index=True)

        data_frame.insert(data_frame.shape[1], "agent", self.agent_name)
        data_frame.insert(data_frame.shape[1], "seed", seed)
        return data_frame

    @ray.method(num_return_vals=1)
    def fft(self, normalize="std", _num_steps=None, _extra_name=""):
        # TODO good if we can restore the weight but not create the agent

        # This line is totally useless. But create dead worker in
        # self.agent.workers.
        # self.agent = restore_agent(self.run_name, self.ckpt, self.env_name)
        self._make_rollout_worker()

        assert self.initialized, "You should reset the worker first!"
        if _num_steps:
            # For testing purpose
            self._num_steps = _num_steps
        if normalize:
            assert isinstance(normalize, str)
            assert normalize in ['range', 'std']
        data_frame = None
        for seed in range(1):
            df = self._efficient_rollout(seed)
            if data_frame is None:
                data_frame = df
            else:
                data_frame = data_frame.append(df, ignore_index=True)

        if self.padding_value is not None:
            data_frame.fillna(self.padding_value, inplace=True)

        label_list = sorted(
            data_frame.label.unique(),
            key=lambda s: int(s[4:]) + (-1e6 if s.startswith('Obs') else +1e6)
        )
        repr_dict = get_representation(
            data_frame, label_list, self.postprocess_func
        )

        # dump the rollout if necessary. This should not REALLY close any
        # worker.
        self.rollout_worker.close()
        ret = (data_frame.copy(), repr_dict)
        return ret


def get_fft_representation(
        name_ckpt_mapping,
        num_seeds,
        num_rollouts,
        padding="fix",
        padding_length=500,
        padding_value=0,
        stack=False,
        normalize="range",
        num_workers=10
):
    initialize_ray()

    data_frame_dict = {}
    representation_dict = {}

    num_agents = len(name_ckpt_mapping)

    num_iteration = int(ceil(num_agents / num_workers))

    agent_ckpt_dict_range = list(name_ckpt_mapping.items())
    agent_count = 1
    agent_count_get = 1

    workers = [FFTWorker.remote() for _ in range(num_workers)]
    now_t_get = now_t = start_t = time.time()

    for iteration in range(num_iteration):
        start = iteration * num_workers
        end = min((iteration + 1) * num_workers, num_agents)
        df_obj_ids = []
        for i, (name, ckpt_dict) in enumerate(agent_ckpt_dict_range[start:end]
                                              ):
            ckpt = ckpt_dict["path"]
            env_name = ckpt_dict["env_name"]
            run_name = ckpt_dict["run_name"]
            env_maker = ENV_MAKER_LOOKUP[env_name]
            workers[i].reset.remote(
                run_name=run_name,
                ckpt=ckpt,
                num_rollouts=num_rollouts,
                env_name=env_name,
                env_maker=env_maker,
                agent_name=name,
                padding=padding,
                padding_length=padding_length,
                padding_value=padding_value,
                worker_name="Worker{}".format(i)
            )

            df_obj_id = workers[i].fft.remote(
                normalize=normalize,
                _extra_name="[{}/{}] ".format(agent_count, num_agents)
            )

            print(
                "[{}/{}] (+{:.1f}s/{:.1f}s) Start collecting data from agent "
                "<{}>".format(
                    agent_count, num_agents,
                    time.time() - now_t,
                    time.time() - start_t, name
                )
            )
            agent_count += 1
            now_t = time.time()
            df_obj_ids.append(df_obj_id)

        for df_obj_id, (name, _) in zip(df_obj_ids,
                                        agent_ckpt_dict_range[start:end]):
            df, rep = copy.deepcopy(ray.get(df_obj_id))
            data_frame_dict[name] = df
            representation_dict[name] = rep
            print(
                "[{}/{}] (+{:.1f}s/{:.1f}s) Got data from agent <{}>".format(
                    agent_count_get, num_agents,
                    time.time() - now_t_get,
                    time.time() - start_t, name
                )
            )
            agent_count_get += 1
            now_t_get = time.time()
    return data_frame_dict, representation_dict


def parse_representation_dict(representation_dict, *args, **kwargs):
    cluster_df = pandas.DataFrame.from_dict(representation_dict).T
    return cluster_df


def get_fft_cluster_finder(
        yaml_path,
        normalize="std",
        try_standardize=False,
        num_agents=None,
        num_seeds=1,
        num_rollouts=100,
        num_workers=10,
        padding="fix",
        padding_length=500,
        padding_value=0,
        show=False,
        num_gpus=0
):
    assert yaml_path.endswith('yaml')
    initialize_ray(num_gpus=num_gpus)

    name_ckpt_mapping = get_name_ckpt_mapping(yaml_path, num_agents)
    print("Successfully loaded name_ckpt_mapping!")

    num_agents = num_agents or len(name_ckpt_mapping)
    # prefix: data/XXX_10agent_100rollout_1seed_28sm29sk
    # /XXX_10agent_100rollout_1seed_28sm29sk
    dir = osp.dirname(yaml_path)
    base = osp.basename(yaml_path)
    prefix = "".join(
        [
            base.split('.yaml')[0], "_{}agents_{}rollout_{}seed_{}".format(
                num_agents, num_rollouts, num_seeds, get_random_string()
            )
        ]
    )
    os.mkdir(osp.join(dir, prefix))
    prefix = osp.join(dir, prefix, prefix)

    data_frame_dict, repr_dict = get_fft_representation(
        name_ckpt_mapping,
        num_seeds,
        num_rollouts,
        normalize=normalize,
        num_workers=num_workers
    )
    print("Successfully get FFT representation!")

    cluster_df = parse_representation_dict(
        repr_dict, padding, padding_length, padding_value
    )
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

    ret = {
        "cluster_finder": {
            'nostd_cluster_finder': nostd_cluster_finder
        },
        "prefix": prefix,
        "cluster_df": cluster_df,
        "data_frame_dict": data_frame_dict,
        "repr_dict": repr_dict
    }

    if try_standardize:
        std_cluster_finder = ClusterFinder(cluster_df, standardize=True)
        std_fig_path = prefix + "_std.png"
        std_cluster_finder.display(save=std_fig_path, show=show)
        print(
            "Successfully finish standardized clustering! Save at: {}".
            format(std_fig_path)
        )
        ret['cluster_finder']["std_cluster_finder"] = std_cluster_finder

    return ret


def test_efficient_rollout():
    initialize_ray(num_gpus=4, test_mode=False)
    num_rollouts = 2
    num_workers = 2
    # yaml_path = "data/0902-ppo-20-agents/0902-ppo-20-agents.yaml"
    yaml_path = "data/0811-random-test.yaml"
    ret = get_fft_cluster_finder(
        yaml_path=yaml_path,
        num_rollouts=num_rollouts,
        num_workers=num_workers,
        num_gpus=4
    )
    return ret


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--yaml-path", type=str, required=True)
    # parser.add_argument("--num-rollouts", type=int, default=100)
    # parser.add_argument("--num-agents", type=int, default=-1)
    # parser.add_argument("--num-workers", type=int, default=10)
    # parser.add_argument("--show", action="store_true", default=False)
    # args = parser.parse_args()
    # initialize_ray(False)
    test_efficient_rollout()

    # ret = get_fft_cluster_finder(
    #     yaml_path=args.yaml_path,
    #     num_agents=None if args.num_agents == -1 else args.num_agents,
    #     num_rollouts=args.num_rollouts,
    #     num_workers=args.num_workers,
    #     # show=args.show
    # )
    # cluster_finder = ret['cluster_finder']['nostd_cluster_finder']
    # prefix = ret['prefix']
    # num_clusters = 10
    # cluster_finder.set(num_clusters)
    # prediction = cluster_finder.predict()
    # print(
    #     "Collected clustering results for {} agents, {} clusters.".format(
    #         len(prediction), num_clusters
    #     )
    # )
    #
    # from process_cluster import load_cluster_df, generate_video_of_cluster
    # from reduce_dimension import reduce_dimension
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    #
    # cluster_df = load_cluster_df(prefix + ".pkl")
    # plot_df, _ = reduce_dimension(cluster_df, prediction, False)
    #
    #
    # def get_label(name):
    #     if name.startswith("ES"):
    #         return "ES"
    #     if "fc2" in name:
    #         return "fc2"
    #     if "fc_out" in name:
    #         return "fc_out"
    #     else:
    #         return "PPO"
    #
    #
    # plot_df.insert(
    #     4, "label", [get_label(name) for name in plot_df.agent], True
    # )
    #
    # save = prefix + "_2d.png"
    # show = None
    #
    #
    # def _get_title(plot_df):
    #     num_clusters = len(plot_df.cluster.unique())
    #     num_agents = len(plot_df.agent.unique())
    #     return "Clustering Result of {} Clusters, " \
    #            "{} Agents (Dimensions Reduced by PCA-TSNE)".format(
    #         num_clusters, num_agents)
    #
    #
    # plt.figure(figsize=(12, 10), dpi=300)
    # num_clusters = len(plot_df.cluster.unique())
    # palette = sns.color_palette(n_colors=num_clusters)
    # ax = sns.scatterplot(
    #     x="x",
    #     y="y",
    #     hue="cluster",
    #     style="label",
    #     palette=palette,
    #     data=plot_df,
    #     legend="full"
    # )
    # ax.set_title(_get_title(plot_df))
    # if save is not None:
    #     assert save.endswith('png')
    #     plt.savefig(save, dpi=300)
    # if show:
    #     plt.show()
    #
    # # generate grid of videos with shape (k, max_num_cols)
    # generate_video_of_cluster(
    #     prediction=prediction,
    #     num_agents=None,
    #     yaml_path=args.yaml_path,
    #     video_prefix=prefix,
    #     max_num_cols=18,
    #     num_workers=args.num_workers
    # )
    # print("Finished generating videos.")
