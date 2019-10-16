import copy
import os
import os.path as osp
import time
from collections import OrderedDict

import IPython
# from IPython.display import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from toolbox.interface.cross_agent import CrossAgentAnalyst
from toolbox.interface.symbolic_agent_rollout import symbolic_agent_rollout
from toolbox.visualize.reduce_dimension import reduce_dimension, draw
from toolbox.visualize.visualize_utils import _generate_gif as generate_gif

# num_agents = 2
# yaml_path = "../data/yaml/0915-halfcheetah-ppo-20-agents.yaml"
# num_rollouts = 2
# num_children = 2
# num_workers = 10
# normal_mean = 1.0
# pca_dim = 3
# std_search_range = np.linspace(0.0, 2, 2)
# dir_name = "./TMP"


num_agents = 20
yaml_path = "../data/yaml/0915-halfcheetah-ppo-20-agents.yaml"
num_rollouts = 10
num_children = 19
num_workers = 16
normal_mean = 1.0
pca_dim = 50
std_search_range = np.linspace(0.0, 2, 51)
dir_name = "../notebooks/1014-scale-distance-relationship-LARGE"

fig_dir_name = osp.join(dir_name, 'fig')

os.makedirs(dir_name, exist_ok=True)
os.makedirs(fig_dir_name, exist_ok=True)

# std_search_range = np.linspace(0.0, 2, 3)
std_ret_dict = {}
std_ret_rollout_dict = {}

now = start = time.time()
# std_load_obj_dict = OrderedDict()


# std_caa_dict = OrderedDict()
std_summary_dict = OrderedDict()
cluster_dataframe = []
joint_cluster_df_dict = OrderedDict()
joint_prediction_dict_dict = OrderedDict()

for i, std in enumerate(std_search_range):
    print("[{}/{}] (+{:.2f}s/{:.2f}s) We are searching std: {}.".format(
        i + 1, len(std_search_range), time.time() - now, time.time() - start,
        std
    ))

    now = time.time()
    rollout_ret, path = symbolic_agent_rollout(
        yaml_path, num_agents, num_rollouts,
        num_workers, num_children,
        std, normal_mean, dir_name
    )

    std_ret_dict[std] = path
    std_ret_rollout_dict[std] = rollout_ret


def remote_restore_and_compute(rollout_ret):

    agent_rollout_dict = OrderedDict()
    name_agent_info_mapping = OrderedDict()
    for key, (rd, ag) in rollout_ret.items():
        agent_rollout_dict[key] = rd
        name_agent_info_mapping[key] = ag

    print("[STD={}] Prepared to create CAA".format(std))

    caa = CrossAgentAnalyst()
    caa.feed(agent_rollout_dict, name_agent_info_mapping)
    print("[STD={}] After feed CAA".format(std))
    result = caa.walkthrough()
    caa.cluster_representation()
    caa.cluster_distance()

    smr = copy.deepcopy(caa.summary())

    print("[STD={}] Collect Summary from CAA".format(std))


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
        d = copy.deepcopy(d)
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
        d = copy.deepcopy(d)
        cluster_dataframe.append(d)

    return copy.deepcopy(caa.cluster_representation_cluster_df_dict), \
           copy.deepcopy(caa.cluster_representation_prediction_dict), \
           copy.deepcopy(cluster_dataframe), \
           copy.deepcopy(smr)


num_workers_caa = 5
remote_restore_and_compute_remote = ray.remote(num_gpus=3.8/num_workers_caa)(remote_restore_and_compute)
obj_id_dict = {}
cluster_dataframe = []

for std, rollout_ret in std_ret_rollout_dict.items():
    obj_id_dict[std] = remote_restore_and_compute_remote(rollout_ret)


    if len(obj_id_dict)==num_workers_caa:
        for std, obj_id in obj_id_dict.items():
            cluster_representation_cluster_df_dict, \
            cluster_representation_prediction_dict, \
            cluster_dataframe_single, \
            summary = copy.deepcopy(ray.get(obj_id))

            joint_cluster_df_dict[std] = copy.deepcopy(
                cluster_representation_cluster_df_dict)

            joint_prediction_dict_dict[std] = copy.deepcopy(
                cluster_representation_prediction_dict
            )
            std_summary_dict[std] = summary

            cluster_dataframe.append(cluster_dataframe_single)
        obj_id_dict.clear()

    for std, obj_id in obj_id_dict.items():
        cluster_representation_cluster_df_dict, \
        cluster_representation_prediction_dict, \
        cluster_dataframe_single, \
        summary = copy.deepcopy(ray.get(obj_id))

        joint_cluster_df_dict[std] = copy.deepcopy(
            cluster_representation_cluster_df_dict)

        joint_prediction_dict_dict[std] = copy.deepcopy(
            cluster_representation_prediction_dict
        )
        std_summary_dict[std] = summary

        cluster_dataframe.append(cluster_dataframe_single)
    # joint_cluster_df = pd.concat(joint_cluster_df)

    # print("[STD={}] Prepared to delete CAA".format(std))
    # del caa

    # print("[STD={}] After delete CAA".format(std))
    print("[{}/{}] (+{:.2f}s/{:.2f}s) Finshed std: {}! Save at: <{}>".format(
        i + 1, len(std_search_range), time.time() - now, time.time() - start,
        std,
        path
    ))

    ray.shutdown()

for std, path in std_ret_dict.items():
    print("Std: {}, Path: {}".format(std, path))

# cluster_dataframe = pd.DataFrame(cluster_dataframe)
#
# dataframe = []
# for std, summary in std_summary_dict.items():
#     data = summary['metric'].copy()
#     for k, v in data.items():
#         if k.endswith('std'):
#             continue
#         dataframe.append({
#             'x': std,
#             'y': v,
#             'label': k
#         })
# labellist = list(data.keys())
# for label in labellist:
#     if label.endswith('std'):
#         continue
#     ylist = [d['y'] for d in dataframe if d['label'] == label]
#     ymin = min(ylist)
#     ymax = max(ylist)
#
#     for d in dataframe:
#         if d['label'] == label:
#             d['y_norm'] = (d['y'] - ymin) / (ymax - ymin + 1e-12)
# dataframe = pd.DataFrame(dataframe)
#
# dash_styles = ["",
#                (4, 1.5),
#                (1, 1),
#                (3, 1, 1.5, 1),
#                (5, 1, 1, 1),
#                (5, 1, 2, 1, 2, 1),
#                (2, 2, 3, 1.5),
#                (1, 2.5, 3, 1.2)] * 3
#
#
# def draw11():
#     plt.figure(figsize=(12, 8), dpi=300)
#     sns.lineplot(x='x', y='y_norm', hue='label', data=dataframe, style='label',
#                  dashes=dash_styles)
#     plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#     plt.title("Scale of STD of Turbulence vs Distance / Reward")
#     plt.savefig(osp.join(fig_dir_name, "std_vs_distance_reward.png"), dpi=300)
#
#
# draw11()
#
#
# # cluster_dataframe = []
# # for std, caa in std_caa_dict.items():
# # precision_dict = caa.cluster_representation_precision_dict
# # parent_cluster_dict = caa.cluster_representation_parent_cluster_dict
# # #     method_precision_dict = {}
# # for (method, predict), accu in zip(parent_cluster_dict.items(),
# #                                    precision_dict.values()):
# #     num_clusters = len(set(predict.values()))
# #     d = {
# #         "num_clusters": num_clusters,
# #         "precision": accu,
# #         "method": "repr." + method,
# #         "std": std
# #     }
# #     cluster_dataframe.append(d)
# #
# # precision_dict = caa.cluster_distance_precision_dict
# # parent_cluster_dict = caa.cluster_distance_parent_cluster_dict
# # #     method_precision_dict = {}
# # for (method, predict), accu in zip(parent_cluster_dict.items(),
# #                                    precision_dict.values()):
# #     num_clusters = len(set(predict.values()))
# #     d = {
# #         "num_clusters": num_clusters,
# #         "precision": accu,
# #         "method": "dist." + method,
# #         "std": std
# #     }
# #     cluster_dataframe.append(d)
#
#
# def draw22():
#     plt.figure(figsize=(12, 8))
#
#     sns.lineplot(data=cluster_dataframe, x='std', y='precision', hue='method',
#                  ci=None, legend=False)
#     sns.scatterplot(data=cluster_dataframe, x='std', y='precision',
#                     hue='method',
#                     size='num_clusters')
#
#     plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#     plt.savefig(osp.join(fig_dir_name, "std_scale_vs_cluster_precision.png"),
#                 dpi=300)
#
#
# draw22()
#
#
# # def display_gif(gif_path):
# #     with open(gif_path, 'rb') as f:
# #         display(IPython.display.Image(data=f.read(), format='png'))
#
#
# def imshow(img, reverse=True):
#     if reverse:
#         img = img[..., ::-1]
#     _, ret = cv2.imencode('.png', img)
#     i = IPython.display.Image(data=ret)
#     IPython.display.display(i)
#
#
# def display_gif(gif_path):
#     with open(gif_path, 'rb') as f:
#         IPython.display.display(
#             IPython.display.Image(data=f.read(), format='png')
#         )
#
#
# def align_plot_df(plot_df):
#     all_cluster = plot_df.cluster.unique()
#
#     parent_index_map = {}
#
#     for i, row in plot_df.iterrows():
#         name = row.agent
#         if name.endswith('child=0'):
#             parent_index_map[row.cluster] = i
#
#     for potential_lost_index in set(all_cluster).difference(
#             set(parent_index_map.keys())):
#         parent_index_map[potential_lost_index] = potential_lost_index + len(
#             plot_df)
#
#     new_cluster_list = []
#     for i, row in plot_df.iterrows():
#         old_cluster = row.cluster
#         new_cluster = parent_index_map[old_cluster]
#         new_cluster_list.append(new_cluster)
#
#     plot_df.cluster = new_cluster_list
#     #         plot_df.iloc[i].cluster = new_cluster
#     #         print('old cluster: {}, new_cluster: {}. Plot df data: {
#     #         }'.format(
#     #             old_cluster, new_cluster, plot_df.iloc[i].cluster))
#     return plot_df
#
#
# def get_figarr(method):
#     std_plot_df_dict = OrderedDict()
#
#     # joint_cluster_df = [caa.cluster_representation_cluster_df_dict[method]
#     #                     for caa in std_caa_dict.values()]
#
#     joint_cluster_df = [
#         dfdict[method] for dfdict in joint_cluster_df_dict.values()
#     ]
#
#     joint_cluster_df = pd.concat(joint_cluster_df)
#
#     precomputed_pca = PCA(pca_dim).fit_transform(joint_cluster_df)
#     precomputed_tsne = TSNE(
#         n_components=2,
#         perplexity=30,
#         verbose=1,
#         random_state=0,
#         n_iter=3000
#     ).fit_transform(precomputed_pca)
#
#     index_count = 0
#
#     for (std, cluster_df), prediction in \
#             zip(joint_cluster_df_dict.items(),
#                 joint_prediction_dict_dict.values()):
#         cluster_df = cluster_df[method]
#         prediction = prediction[method]
#         # cluster_df = caa.cluster_representation_cluster_df_dict[method]
#         # prediction = caa.cluster_representation_prediction_dict[method]
#
#         interval = cluster_df.shape[0]
#
#         plot_df = reduce_dimension(
#             cluster_df, prediction, computed_embedding=precomputed_tsne[
#                                                        index_count:
#                                                        index_count + interval]
#         )[0]
#
#         index_count += interval
#
#         # align plot_df
#         plot_df = align_plot_df(plot_df)
#
#         std_plot_df_dict[std] = plot_df
#
#     xmax = max([plot_df.x.max() for plot_df in std_plot_df_dict.values()])
#     xmin = min([plot_df.x.min() for plot_df in std_plot_df_dict.values()])
#
#     ymax = max([plot_df.y.max() for plot_df in std_plot_df_dict.values()])
#     ymin = min([plot_df.y.min() for plot_df in std_plot_df_dict.values()])
#     figarr_list = []
#     for std, plot_df in std_plot_df_dict.items():
#         figarr = draw(
#             plot_df, return_array=True, dpi=72,
#             title="STD={:.3f}".format(std),
#             xlim=(xmin, xmax), ylim=(ymin, ymax),
#             emphasis_parent=True
#         )
#         figarr_list.append(figarr)
#     return figarr_list
#
#
# for method in ['fft', 'naive', 'fft_pca', 'naive_pca']:
#     figarr_list = get_figarr(method)
#     generate_gif(np.stack(figarr_list),
#                  osp.join(fig_dir_name, "ani_{}.gif".format(method)), fps=1)
