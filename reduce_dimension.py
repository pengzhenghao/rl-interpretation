"""
This code is copied from uber atari-model-zoo projects. Link:

https://github.com/uber-research/atari-model-zoo/blob/master/
dimensionality_reduction/process_helper.py
"""
import matplotlib.pyplot as plt
import pandas
import seaborn as sns
from sklearn import decomposition, manifold

DEFAULT_METHOD = {
    "name": "pca_tsne",
    "pca_dim": 50,
    "perplexity": 30,
    "n_iter": 3000
}


def reduce_dimension(data, prediction, three_dimensional=False):
    method = DEFAULT_METHOD
    perplexity = method['perplexity']
    n_iter = method['n_iter']
    pca_dim = method['pca_dim']
    tsne_dim = 3 if three_dimensional else 2

    print('Running pca')
    pca_result = decomposition.PCA(pca_dim).fit_transform(data)

    print('Running tsne')
    result = manifold.TSNE(
        n_components=tsne_dim,
        perplexity=perplexity,
        verbose=2,
        random_state=0,
        n_iter=n_iter
    ).fit_transform(pca_result)
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

    result_df = pandas.DataFrame(
        [
            get_row(name, coord)
            for name, coord in zip(prediction.keys(), result)
        ]
    )
    return result_df, result


def draw(plot_df, show=True, save=None):
    three_dimensional = 'z' in plot_df.columns
    if three_dimensional:
        _draw_3d(plot_df, show, save)
    else:
        _draw_2d(plot_df, show, save)
    print("Drew!")


def _draw_2d(plot_df, show=True, save=None):
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
    if save is not None:
        assert save.endswith('png')
        plt.savefig(save, dpi=300)
    if show:
        plt.show()


def _draw_3d(plot_df, show=True, save=None):
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

    ax.set_title(_get_title(plot_df))
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
