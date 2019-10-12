"""
This code is copied from uber atari-model-zoo projects. Link:

https://github.com/uber-research/atari-model-zoo/blob/master/
dimensionality_reduction/process_helper.py
"""
import matplotlib.pyplot as plt
import pandas
import seaborn as sns
from sklearn import decomposition, manifold
import numpy as np
DEFAULT_METHOD = {
    "name": "pca_tsne",
    "pca_dim": 50,
    "perplexity": 30,
    "n_iter": 3000
}


def reduce_dimension(data, prediction, three_dimensional=False, pca_dim=None,
                     precomputed_pca=None, computed_embedding=None):
    """

    :param data: dataframe with shape [num_agents, num_features]
    :param prediction: {name: {cluster: int, ..}, ..}
    :param three_dimensional:
    :return:
    """
    method = DEFAULT_METHOD
    perplexity = method['perplexity']
    n_iter = method['n_iter']
    pca_dim = method['pca_dim'] if pca_dim is None else pca_dim
    tsne_dim = 3 if three_dimensional else 2

    if computed_embedding is None:
        print('Running pca')
        if precomputed_pca is None:
            pca_result = decomposition.PCA(pca_dim).fit_transform(data)
        else:
            assert isinstance(precomputed_pca, decomposition.PCA)
            print("Detected precomputed PCA instance! "
                  "We will use it to conduct dimension reduction!")
            pca_result = precomputed_pca.transform(data)

        print('Running tsne')
        result = manifold.TSNE(
            n_components=tsne_dim,
            perplexity=perplexity,
            verbose=2,
            random_state=0,
            n_iter=n_iter
        ).fit_transform(pca_result)
    else:
        result = computed_embedding

    assert data.shape[0] == result.shape[0]
    print(
        'Reduction Completed! data.shape={} result.shape={}'.format(
            data.shape, result.shape
        )
    )

    def get_row(name, coordinates):
        ret = {
            "agent": name,
            "x": coordinates[0],
            "y": coordinates[1],
            "cluster": prediction[name]['cluster']
        }
        if three_dimensional:
            ret['z'] = coordinates[2]
        return ret

    plot_df = pandas.DataFrame(
        [
            get_row(name, coord)
            for name, coord in zip(prediction.keys(), result)
        ]
    )
    return plot_df, result


def draw(plot_df, show=True, save=None, title=None, return_array=False, dpi=300, **kwargs):
    three_dimensional = 'z' in plot_df.columns
    if three_dimensional:
        return _draw_3d(plot_df, show, save, title, return_array, dpi)
    else:
        return _draw_2d(plot_df, show, save, title, return_array, dpi, **kwargs)
    # print("Drew!")


def _draw_2d(plot_df, show=True, save=None, title=None, return_array=False, dpi=300, xlim=None, ylim=None, **kwargs):
    fig = plt.figure(figsize=(12, 10), dpi=dpi)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    num_clusters = len(plot_df.cluster.unique())
    palette = sns.color_palette(n_colors=num_clusters)
    ax = sns.scatterplot(
        x="x",
        y="y",
        hue="cluster",
        style="cluster",
        palette=palette,
        data=plot_df,
        legend="full"
    )

    if kwargs['emphasis_parent']:
        emphasis_parent = plot_df.copy()

        flag_list = []
        for name in emphasis_parent.agent:
            flag = name.endswith('child=0')
            flag_list.append(flag)

        emphasis_parent = emphasis_parent[flag_list]

        palette = sns.color_palette(
            n_colors=len(emphasis_parent.cluster.unique())
        )

        # print('len platter: {}, len parents {}'.format(len(palette), len(emphasis_parent)))

        # palette = sns.color_palette(n_colors=len(emphasis_parent.cluster.unique()))
        ax = sns.scatterplot(
            x="x",
            y="y",
            hue="cluster",
            style="cluster",
            s=200,
            palette=palette,
            data=emphasis_parent,
            legend=False,
            ax=ax
        )


    title = "[{}]".format(title) if title is not None else ""
    ax.set_title(title + _get_title(plot_df))
    if save is not None:
        assert save.endswith('png')
        plt.savefig(save, dpi=300)
    if show:
        plt.show()
    if return_array:
        # fig = plt.gcf()
        fig.canvas.draw()
        figarr = np.array(fig.canvas.renderer.buffer_rgba())[..., [2, 1, 0, 3]]
        return figarr


def _draw_3d(plot_df, show=True, save=None, title=None):
    from mpl_toolkits.mplot3d import Axes3D
    use_less_var = Axes3D

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
    title = "[{}]".format(title) if title is not None else ""
    ax.set_title(title + _get_title(plot_df))
    if save is not None:
        assert save.endswith('png')
        plt.savefig(save, dpi=300)
    if show:
        plt.show()


def _get_title(plot_df):
    num_clusters = len(plot_df.cluster.unique())
    num_agents = len(plot_df.agent.unique())
    return "Clustering Result of {} Clusters, " \
           "{} Agents (Dimensions Reduced by PCA-TSNE)".format(
        num_clusters, num_agents)
