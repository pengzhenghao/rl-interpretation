import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


def cluster(cluster_df, search_range):
    kmeans = [KMeans(n_clusters=i) for i in search_range]
    fit_result = [kmeans[i].fit(cluster_df) for i in range(len(kmeans))]
    return fit_result


def score(cluster_df, fit_list):
    score = [fit.score(cluster_df) for fit in fit_list]
    cost = -np.asarray(score)
    return cost


def predict(cluster_df, kmeans):
    ret = kmeans.predict(cluster_df)
    assert len(cluster_df) == len(ret)
    return ret


def display(search_range, cost, log=True):
    process = np.log if log else lambda x: x
    # search_range = range(1, max_num_cluster)
    plt.plot(search_range, process(cost))
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title('Cost [Elbow Curve] (Processed by log(x))')
    plt.show()


class ClusterFinder(object):
    def __init__(self, cluster_df, max_num_cluster):
        assert cluster_df.ndim == 2
        self.cluster_df = cluster_df
        self.best_k = None
        self.max_num_cluster = max_num_cluster
        self.search_range = range(1, self.max_num_cluster)
        self.fits = cluster(self.cluster_df, self.search_range)
        print(
            "Clustering Finished! Call ClusterFinder.display to see "
            "the elbow curve."
        )

    def predict(self):
        assert self.best_k is not None, "Call ClusterFinder.set(k) to set " \
                                        "the best number of cluster."
        return predict(self.cluster_df, self.fits[self.best_k])

    def set(self, k):
        self.best_k = k

    def display(self):
        cost = score(self.cluster_df, self.fits)
        display(self.search_range, cost)
