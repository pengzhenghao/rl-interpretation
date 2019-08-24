import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def score(cluster_df, fit_list):
    score = [fit.score(cluster_df) for fit in fit_list.values()]
    cost = -np.asarray(score)
    cost = {k: cost[i] for i, k in enumerate(fit_list.keys())}
    return cost


def display(search_range, cost, log=True, save=False, show=True):
    process = np.log if log else lambda x: x
    plt.figure(figsize=(np.sqrt(max(search_range)) + 10, 10))

    cost_list = [cost[i] for i in search_range]

    plt.plot(search_range, process(cost_list))
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.xticks(list(search_range), [str(t) for t in search_range])
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


def reduce_dimension(data_frame):
    assert data_frame.ndim == 2
    return data_frame


class ClusterFinder(object):
    def __init__(self, cluster_df, max_num_cluster=None, standardize=False):
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
        cost = score(self.cluster_df, self.fits)
        if save:
            assert isinstance(save, str)
            assert save.endswith('png')
        display(self.search_range, cost, log, save, show)

    def visualize(self, three_dimension=False):
        # TODO
        # Should show the 2D or 3D embedding of the representation.
        reduced = reduce_dimension(self.cluster_df)
        assert reduced.ndim == 2
        assert reduced.shape[1] == (3 if three_dimension else 2)
