"""
This code is copied from uber atari-model-zoo projects. Link:

https://github.com/uber-research/atari-model-zoo/blob/master/
dimensionality_reduction/process_helper.py
"""

import matplotlib.pyplot as plt
import pandas
import seaborn as sns
# from collections import OrderedDict
from sklearn import decomposition, manifold

# import atari_zoo
# from atari_zoo import MakeAtariModel
# import urllib.request
# import os
# method= {
#     "name": "pca_tsne",
#     "pca_dim": 50,
#     "perplexity": 30,
#     "n_iter": 3000
# }

DEFAULT_METHOD = {
    "name": "pca_tsne",
    "pca_dim": 50,
    "perplexity": 30,
    "n_iter": 1000,
    "tsne_dim": 2
}


def reduce_dimension(data, prediction, three_dimensional):
    # print(method)

    print("Reducing Dimensions")
    method = DEFAULT_METHOD
    perplexity = method['perplexity']
    n_iter = method['n_iter']
    pca_dim = method['pca_dim']
    tsne_dim = 3 if three_dimensional else 2
    # if 'tsne' in method['name']:
    #     if 'perplexity' in method:
    #         perplexity = method['perplexity']
    #     if 'n_iter' in method:
    #         n_iter = method['n_iter']
    #     if 'pca_dim' in method:
    #         pca_dim = method['pca_dim']
    #     if 'tsne_dim' in method:
    #         tsne_dim = method['tsne_dim']

    # print(perplexity, n_iter, pca_dim)
    # if method['name'] == 'tsne':
    #     print('running tsne')
    #     result = manifold.TSNE(
    #         n_components=2,
    #         perplexity=perplexity,
    #         verbose=2,
    #         random_state=0,
    #         n_iter=n_iter
    #     ).fit_transform(data)
    # elif method['name'] == 'pca':
    #     print('running pca')
    #     result = decomposition.PCA(2).fit_transform(data)
    if method['name'] == 'pca_tsne':
        print('running pca')
        pca_result = decomposition.PCA(pca_dim).fit_transform(data)
        print('running tsne')
        result = manifold.TSNE(
            n_components=tsne_dim,
            perplexity=perplexity,
            verbose=2,
            random_state=0,
            n_iter=n_iter
        ).fit_transform(pca_result)
    # elif method['name'] == 'debug':
    #     print('running debug')
    #     nrow, ncol = data.shape
    #     idx_last_x, idx_last_y = int(ncol / 2 - 1), -1
    #     result = np.hstack(
    #         (
    #             data[:, idx_last_x].reshape(nrow, 1),
    #             data[:, idx_last_y].reshape(nrow, 1)
    #         )
    #     )
    else:
        raise NotImplementedError

    print(
        'Reduction Completed! data.shape={} result.shape={}'.format(
            data.shape, result.shape
        )
    )

    def get_row(name, coordinates):
        if three_dimensional:
            ret = {
                "agent": name,
                "x": coordinates[0],
                "y": coordinates[1],
                "z": coordinates[2],
                "cluster": prediction[name]['cluster']
            }
        else:
            ret = {
                "agent": name,
                "x": coordinates[0],
                "y": coordinates[1],
                "cluster": prediction[name]['cluster']
            }
        return ret

    result_df = pandas.DataFrame(
        [
            get_row(name, coord)
            for name, coord in zip(prediction.index, result)
        ]
    )
    return result_df, result


def display(plot_df):
    three_dimensional = 'z' in plot_df.columns
    if three_dimensional:
        _draw_3d(plot_df)
    else:
        _draw_2d(plot_df)
    print("Drew!")


def _draw_2d(plot_df):
    plt.figure(figsize=(12, 10), dpi=300)
    num_clusters = len(plot_df.cluster.unique())
    palette = sns.color_palette(n_colors=num_clusters)
    ax = sns.scatterplot(
        x="x",
        y="y",
        hue="cluster",
        palette=palette,
        data=plot_df,
        legend="full"
    )
    ax.set_title(_get_title(plot_df))


def _draw_3d(plot_df):
    from mpl_toolkits.mplot3d import Axes3D
    Axes3D()

    fig = plt.figure(figsize=(12, 10), dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    labels = plot_df.cluster.unique()
    labels = sorted(labels)
    groupby = plot_df.groupby("cluster")

    for label in labels:
        d = groupby.get_group(label)
        ax.scatter(d.x, d.y, d.z, label="Cluster {}".format(label))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    ax.set_title(_get_title(plot_df))


def _get_title(plot_df):
    num_clusters = len(plot_df.cluster.unique())
    num_agents = len(plot_df.agent.unique())
    return "Clustering Result of {} Clusters, " \
           "{} Agents (Dimensions Reduced by PCA-tSNE)".format(
        num_clusters, num_agents)


def test():
    from process_cluster import load_cluster_df
    cluster_df = load_cluster_df("")


if __name__ == '__main__':
    test()
