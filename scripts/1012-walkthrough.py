import time
import uuid
from collections import OrderedDict

import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from toolbox.interface.cross_agent import CrossAgentAnalyst
from toolbox.interface.symbolic_agent_rollout import symbolic_agent_rollout
from toolbox.visualize.reduce_dimension import reduce_dimension, draw
from toolbox.visualize.visualize_utils import _generate_gif as generate_gif

# from IPython.display import Image
import cv2
import IPython
import os

num_agents = 20
yaml_path = "../data/yaml/0915-halfcheetah-ppo-20-agents.yaml"
num_rollouts = 10
num_children = 19
num_workers = 10

# normal_std = 0.1
normal_mean = 1.0

spawn_seed = 0
num_samples = 200  # From each agent's dataset
pca_dim = 50


# dir_name = "./"
dir_name = "../notebooks/1013-scale-distance-relationship-LARGE"
fig_dir_name = osp.join(dir_name, 'fig')

os.makedirs(dir_name, exist_ok=True)
os.makedirs(fig_dir_name, exist_ok=True)

std_search_range = np.linspace(0.0, 2, 61)
# std_search_range = np.linspace(0.0, 2, 3)
std_ret_dict = {}

now = start = time.time()
std_load_obj_dict = OrderedDict()

for i, std in enumerate(std_search_range):
    print("[{}/{}] (+{:.2f}s/{:.2f}s) We are searching std: {}.".format(
        i + 1, len(std_search_range), time.time() - now, time.time() - start,
        std
    ))
    now = time.time()
    a, path = symbolic_agent_rollout(
        yaml_path, num_agents, num_rollouts,
        num_workers, num_children,
        std, normal_mean, dir_name
    )

    std_load_obj_dict[std] = a

    print("[{}/{}] (+{:.2f}s/{:.2f}s) Finshed std: {}! Save at: <{}>".format(
        i + 1, len(std_search_range), time.time() - now, time.time() - start,
        std,
        path
    ))
    std_ret_dict[std] = path

for std, path in std_ret_dict.items():
    print("Std: {}, Path: {}".format(std, path))



std_caa_dict = OrderedDict()
std_summary_dict = OrderedDict()

now = start = time.time()

for i, (std, load_obj) in enumerate(std_load_obj_dict.items()):

    print("[{}/{}] (+{:.2f}s/{:.2f}s) Current std: {}.".format(
        i + 1, len(std_load_obj_dict), time.time() - now, time.time() - start,
        std
    ))
    now = time.time()

    rollout_ret = load_obj

    agent_rollout_dict = OrderedDict()
    name_agent_info_mapping = OrderedDict()
    for key, (rd, ag) in rollout_ret.items():
        agent_rollout_dict[key] = rd
        name_agent_info_mapping[key] = ag

    caa = CrossAgentAnalyst()
    caa.feed(agent_rollout_dict, name_agent_info_mapping)
    result = caa.walkthrough()
    prs, prd, pac, cdf = caa.cluster_representation()
    ret = caa.cluster_distance()
    smr = caa.summary()

    std_caa_dict[std] = caa
    std_summary_dict[std] = smr

dataframe = []

for std, summary in std_summary_dict.items():
    data = summary['metric'].copy()
    for k, v in data.items():
        if k.endswith('std'):
            continue
        dataframe.append({
            'x': std,
            'y': v,
            'label': k
        })

labellist = list(data.keys())

for label in labellist:
    if label.endswith('std'):
        continue
    ylist = [d['y'] for d in dataframe if d['label'] == label]
    ymin = min(ylist)
    ymax = max(ylist)

    for d in dataframe:
        if d['label'] == label:
            d['y_norm'] = (d['y'] - ymin) / (ymax - ymin + 1e-12)

dataframe = pd.DataFrame(dataframe)

dash_styles = ["",
               (4, 1.5),
               (1, 1),
               (3, 1, 1.5, 1),
               (5, 1, 1, 1),
               (5, 1, 2, 1, 2, 1),
               (2, 2, 3, 1.5),
               (1, 2.5, 3, 1.2)] * 3

def draw11():
    plt.figure(figsize=(12, 8), dpi=300)
    sns.lineplot(x='x', y='y_norm', hue='label', data=dataframe, style='label',
                 dashes=dash_styles)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title("Scale of STD of Turbulence vs Distance / Reward")
    plt.savefig(osp.join(fig_dir_name, "std_vs_distance_reward.png"), dpi=300)

draw11()

cluster_dataframe = []
for std, caa in std_caa_dict.items():
    precision_dict = caa.cluster_representation_precision_dict
    parent_cluster_dict = caa.cluster_representation_parent_cluster_dict
    #     method_precision_dict = {}
    for (method, predict), accu in zip(parent_cluster_dict.items(),
                                       precision_dict.values()):
        num_clusters = len(set(predict.values()))
        d = {
            "num_clusters": num_clusters,
            "precision": accu,
            "method": "repr." + method,
            "std": std
        }
        cluster_dataframe.append(d)

    precision_dict = caa.cluster_distance_precision_dict
    parent_cluster_dict = caa.cluster_distance_parent_cluster_dict
    #     method_precision_dict = {}
    for (method, predict), accu in zip(parent_cluster_dict.items(),
                                       precision_dict.values()):
        num_clusters = len(set(predict.values()))
        d = {
            "num_clusters": num_clusters,
            "precision": accu,
            "method": "dist." + method,
            "std": std
        }
        cluster_dataframe.append(d)
#     method_precision_dict[method] = [num_clusters, accu]

cluster_dataframe = pd.DataFrame(cluster_dataframe)


def draw22():
    plt.figure(figsize=(12, 8))

    sns.lineplot(data=cluster_dataframe, x='std', y='precision', hue='method',
                 ci=None, legend=False)
    sns.scatterplot(data=cluster_dataframe, x='std', y='precision', hue='method',
                    size='num_clusters')

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(osp.join(fig_dir_name, "std_scale_vs_cluster_precision.png"), dpi=300)

draw22()

# def display_gif(gif_path):
#     with open(gif_path, 'rb') as f:
#         display(IPython.display.Image(data=f.read(), format='png'))


def imshow(img, reverse=True):
    if reverse:
        img = img[..., ::-1]
    _, ret = cv2.imencode('.png', img)
    i = IPython.display.Image(data=ret)
    IPython.display.display(i)


def display_gif(gif_path):
    with open(gif_path, 'rb') as f:
        IPython.display.display(
            IPython.display.Image(data=f.read(), format='png')
        )


def align_plot_df(plot_df):
    all_cluster = plot_df.cluster.unique()

    parent_index_map = {}

    for i, row in plot_df.iterrows():
        name = row.agent
        if name.endswith('child=0'):
            parent_index_map[row.cluster] = i

    for potential_lost_index in set(all_cluster).difference(
            set(parent_index_map.keys())):
        parent_index_map[potential_lost_index] = potential_lost_index + len(
            plot_df)

    new_cluster_list = []
    for i, row in plot_df.iterrows():
        old_cluster = row.cluster
        new_cluster = parent_index_map[old_cluster]
        new_cluster_list.append(new_cluster)

    plot_df.cluster = new_cluster_list
    #         plot_df.iloc[i].cluster = new_cluster
    #         print('old cluster: {}, new_cluster: {}. Plot df data: {
    #         }'.format(
    #             old_cluster, new_cluster, plot_df.iloc[i].cluster))
    return plot_df


def get_figarr(method):
    std_plot_df_dict = OrderedDict()

    joint_cluster_df = [caa.cluster_representation_cluster_df_dict[method]
                        for caa in std_caa_dict.values()]
    joint_cluster_df = pd.concat(joint_cluster_df)

    precomputed_pca = PCA(pca_dim).fit_transform(joint_cluster_df)
    precomputed_tsne = TSNE(
        n_components=2,
        perplexity=30,
        verbose=1,
        random_state=0,
        n_iter=3000
    ).fit_transform(precomputed_pca)

    index_count = 0

    for std, caa in std_caa_dict.items():
        cluster_df = caa.cluster_representation_cluster_df_dict[method]
        prediction = caa.cluster_representation_prediction_dict[method]

        interval = cluster_df.shape[0]

        plot_df = reduce_dimension(
            cluster_df, prediction, computed_embedding=precomputed_tsne[
                                                       index_count:
                                                       index_count + interval]
        )[0]

        index_count += interval

        # align plot_df
        plot_df = align_plot_df(plot_df)

        std_plot_df_dict[std] = plot_df

    xmax = max([plot_df.x.max() for plot_df in std_plot_df_dict.values()])
    xmin = min([plot_df.x.min() for plot_df in std_plot_df_dict.values()])

    ymax = max([plot_df.y.max() for plot_df in std_plot_df_dict.values()])
    ymin = min([plot_df.y.min() for plot_df in std_plot_df_dict.values()])
    figarr_list = []
    for std, plot_df in std_plot_df_dict.items():
        figarr = draw(
            plot_df, return_array=True, dpi=72,
            title="STD={:.3f}".format(std),
            xlim=(xmin, xmax), ylim=(ymin, ymax),
            emphasis_parent=True
        )
        figarr_list.append(figarr)
    return figarr_list


for method in ['fft', 'naive', 'fft_pca', 'naive_pca']:
    figarr_list = get_figarr(method)
    generate_gif(np.stack(figarr_list), osp.join(fig_dir_name, "ani_{}.gif".format(method)), fps=1)
