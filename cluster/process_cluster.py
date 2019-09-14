import argparse
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# from record_video import generate_video_of_cluster
from visualize.reduce_dimension import reduce_dimension, draw


class ClusterFinder(object):
    def __init__(self, cluster_df, max_num_cluster=100, standardize=False):
        # We argue that the max cluster num larger than 100 is meaningless
        assert cluster_df.ndim == 2
        self.keys = cluster_df.index
        if standardize:
            standardized_df = StandardScaler().fit_transform(cluster_df)
            self.standardized = True
        else:
            standardized_df = cluster_df
            self.standardized = False
        self.cluster_df = standardized_df
        self.best_k = None
        self.max_num_cluster = max_num_cluster or len(cluster_df)
        self.search_range = range(1, self.max_num_cluster + 1)
        self.initialized = False
        self.fits = None

    def _fit_all(self):
        if self.initialized:
            assert self.fits is not None
            return self.fits
        kmeans = {i: KMeans(n_clusters=i) for i in self.search_range}
        fit_result = {
            i: kmeans[i].fit(self.cluster_df)
            for i in self.search_range
        }
        self.fits = fit_result
        print(
            "Clustering Finished! Call ClusterFinder.display to see "
            "the elbow curve."
        )
        self.initialized = True
        return self.fits

    def predict(self):
        """
        Return a dict whose key are names of agents and values is the cluster
        indices.
        :return:
        """
        assert self.best_k is not None, "Call ClusterFinder.set(k) to set " \
                                        "the best number of cluster."
        if not self.initialized:
            assert self.fits is None
            self.fits = dict()
            self.fits[self.best_k] = KMeans(n_clusters=self.best_k
                                            ).fit(self.cluster_df)

        assert self.best_k in self.fits
        prediction = self.fits[self.best_k].predict(self.cluster_df)
        distances = self.fits[self.best_k].transform(self.cluster_df)
        ret = {}
        for (index, name), pred in zip(enumerate(self.keys), prediction):
            dis = distances[index, pred]
            info = {"distance": dis, "cluster": pred, "name": name}
            ret[name] = info
        return ret

    def set(self, k):
        self.best_k = k

    def display(self, log=False, save=False, show=True):
        self._fit_all()

        score = [fit.score(self.cluster_df) for fit in self.fits.values()]
        cost = -np.asarray(score)
        cost = {k: cost[i] for i, k in enumerate(self.fits.keys())}

        if save:
            assert isinstance(save, str)
            assert save.endswith('png')
        process = np.log if log else lambda x: x
        plt.figure(figsize=(np.sqrt(max(self.search_range)) + 10, 10))

        cost_list = [cost[i] for i in self.search_range]

        plt.plot(self.search_range, process(cost_list))
        plt.xlabel('Number of Clusters')
        plt.ylabel('Score')
        plt.xticks(
            list(self.search_range), [str(t) for t in self.search_range]
        )
        if log:
            plt.title('Cost [Elbow Curve] (Processed by log(x))')
        else:
            plt.title('Cost [Elbow Curve]')
        plt.grid()
        if save:
            assert isinstance(save, str)
            assert save.endswith('png')
            plt.savefig(save, dpi=300)
        if show:
            plt.show()


def load_cluster_df(pkl_path):
    assert pkl_path.endswith(".pkl")
    assert osp.exists(pkl_path)
    cluster_df = pandas.read_pickle(pkl_path)
    return cluster_df


if __name__ == '__main__':
    """
    Must given the number of cluster (--num-clusters or -k).
    Generate videos, visualization figures for the clustering results.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--yaml-path", type=str, required=True)
    parser.add_argument("--num-clusters", "-k", type=int, required=True)
    parser.add_argument("--run-name", type=str, default="PPO")
    parser.add_argument("--env-name", type=str, default="BipedalWalker-v2")
    parser.add_argument("--num-rollouts", type=int, default=100)
    parser.add_argument("--num-agents", type=int, default=-1)
    parser.add_argument("--show", action="store_true", default=False)
    parser.add_argument("--max-num-cols", type=int, default=11)
    parser.add_argument("--num-workers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    yaml_path = args.yaml_path
    cluster_df_path = args.root + ".pkl"
    assert osp.exists(osp.dirname(args.root)), osp.dirname(args.root)
    assert yaml_path.endswith(".yaml")
    assert osp.exists(yaml_path), yaml_path
    assert osp.exists(cluster_df_path), cluster_df_path
    prefix = args.root

    # load cluster_df
    cluster_df = load_cluster_df(cluster_df_path)
    print(
        "Loaded cluster data frame from <{}> whose shape is {}.".format(
            cluster_df_path, cluster_df.shape
        )
    )

    num_agents = args.num_agents if args.num_agents != -1 else len(cluster_df)
    assert 1 <= args.num_clusters <= num_agents, (
        args.num_clusters, len(cluster_df)
    )

    # get clustering result
    cluster_finder = ClusterFinder(cluster_df)
    cluster_finder.set(args.num_clusters)
    prediction = cluster_finder.predict()
    print(
        "Collected clustering results for {} agents, {} clusters.".format(
            len(prediction), args.num_clusters
        )
    )

    df_2d, _ = reduce_dimension(cluster_df, prediction, False)
    draw(df_2d, show=False, save=prefix + "_2d.png")

    df_3d, _ = reduce_dimension(cluster_df, prediction, True)
    draw(df_3d, show=False, save=prefix + "_3d.png")
    print("Figures have been saved at {}_**.png".format(prefix))

    # generate grid of videos with shape (k, max_num_cols)
    # generate_video_of_cluster(
    #     prediction=prediction,
    #     num_agents=num_agents,
    #     yaml_path=yaml_path,
    #     video_prefix=prefix,
    #     seed=args.seed,
    #     max_num_cols=args.max_num_cols,
    #     num_workers=args.num_workers
    # )
    # print("Finished generating videos.")
