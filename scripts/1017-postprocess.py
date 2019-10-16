import copy
import os
import os.path as osp
import pickle
import time
from collections import OrderedDict

from toolbox.utils import initialize_ray

dir_name = "./1014-scale-distance-relationship-LARGE"

std_pkl_dict = []
for pkl_file in os.listdir(dir_name):
    if not pkl_file.endswith(".pkl"):
        continue
    std = eval(pkl_file.split("_")[-1].split('std')[0])
    std_pkl_dict.append([std, pkl_file])

std_pkl_dict = sorted(std_pkl_dict, key=lambda x: x[0])

from toolbox.interface.cross_agent import CrossAgentAnalyst

initialize_ray(num_gpus=4, test_mode=False, object_store_memory=40 * int(1e9))

# std_load_obj_dict = OrderedDict()
# std_summary_dict = OrderedDict()
# cluster_dataframe = []
# joint_cluster_df_dict = OrderedDict()
# joint_prediction_dict_dict = OrderedDict()

tt = time.time


def remote_restore_and_compute(pkl_file, now, start):
    cluster_dataframe_inside = []
    file_name = osp.join(dir_name, pkl_file)

    print("(+{:.2f}s/{:.2f}s) Current std: {}. Start to pickle <{}>".format(
        time.time() - now, time.time() - start, std, file_name
    ))
    now = time.time()
    with open(file_name, 'rb') as f:
        rollout_ret = pickle.load(f)
    print("(+{:.2f}s/{:.2f}s) Current std: {}. Finish to pickle <{}>".format(
        time.time() - now, time.time() - start, std, file_name
    ))
    now = time.time()

    agent_rollout_dict = OrderedDict()
    name_agent_info_mapping = OrderedDict()
    for key, (rd, ag) in rollout_ret.items():
        agent_rollout_dict[key] = rd
        name_agent_info_mapping[key] = ag

    print("[STD={}] (+{:.2f}s/{:.2f}s) Prepared to create CAA".format(
        std,
        tt() -
        now,
        tt() -
        start))
    now = tt()
    caa = CrossAgentAnalyst()
    caa.feed(agent_rollout_dict, name_agent_info_mapping)

    print("[STD={}] (+{:.2f}s/{:.2f}s) Start to walkthrough CAA".format(
        std,
        tt()
        - now,
        tt()
        -
        start))
    now = tt()
    result = caa.walkthrough()

    print(
        "[STD={}] (+{:.2f}s/{:.2f}s) Start to collect representation "
        "CAA".format(
            std,
            tt() - now, tt() - start))
    now = tt()
    caa.cluster_representation()

    print("[STD={}] (+{:.2f}s/{:.2f}s) Start to cal distances CAA".format(
        std,
        tt() - now,
        tt() - start))
    now = tt()
    caa.cluster_distance()

    print(
        "[STD={}] (+{:.2f}s/{:.2f}s) Start to collect summary CAA".format(
            std,
            tt() - now,
            tt() - start))
    now = tt()
    smr = copy.deepcopy(caa.summary())

    print(
        "[STD={}] (+{:.2f}s/{:.2f}s) Start to collect clutser_dataframe "
        "CAA".format(
            std, tt() - now, tt() - start))
    now = tt()
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
        cluster_dataframe_inside.append(d)

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
        cluster_dataframe_inside.append(d)
    print(
        "[STD={}] (+{:.2f}s/{:.2f}s) Start to return from remote function "
        "CAA".format(
            std, tt() - now, tt() - start))
    now = tt()
    return copy.deepcopy(caa.cluster_representation_cluster_df_dict), \
           copy.deepcopy(caa.cluster_representation_prediction_dict), \
           copy.deepcopy(cluster_dataframe_inside), \
           copy.deepcopy(smr)


# num_workers_caa = 13
# remote_restore_and_compute_remote = ray.remote(
# num_gpus=3.8/num_workers_caa)(remote_restore_and_compute)
# obj_id_dict = {}
# cluster_dataframe = []

# std_load_obj_dict = OrderedDict()
std_summary_dict = OrderedDict()
cluster_dataframe = []
joint_cluster_df_dict = OrderedDict()
joint_prediction_dict_dict = OrderedDict()

now = start = time.time()
print_count = 1
print("start to load")

for i, (std, pkl_file) in enumerate(std_pkl_dict):
    print("[{}/{}] current file: ".format(i + 1, len(std_pkl_dict)), std,
          pkl_file)

    cluster_representation_cluster_df_dict, \
    cluster_representation_prediction_dict, \
    cluster_dataframe_single, \
    summary = remote_restore_and_compute(
        pkl_file, now, start
    )
    joint_cluster_df_dict[std] = cluster_representation_cluster_df_dict
    joint_prediction_dict_dict[std] = cluster_representation_prediction_dict
    std_summary_dict[std] = summary
    cluster_dataframe.append(cluster_dataframe_single)
    print("[{}/{}] (+{:.2f}s/{:.2f}s) Finshed std: {}!".format(
        i + 1, len(std_pkl_dict), time.time() - now, time.time() - start,
        std
    ))
    now = time.time()

    ckpt_path_name = osp.join(dir_name, "CAA_result_ckpt{}.pkl".format(i))
    with open(ckpt_path_name, 'wb') as f:
        pickle.dump(
            [joint_cluster_df_dict, joint_prediction_dict_dict,
             std_summary_dict, cluster_dataframe], f
        )
        print("ckpt is dump at: ", ckpt_path_name)

ckpt_path_name = osp.join(dir_name, "CAA_result_final.pkl")
with open(ckpt_path_name, 'wb') as f:
    pickle.dump(
        [joint_cluster_df_dict, joint_prediction_dict_dict, std_summary_dict,
         cluster_dataframe], f
    )
    print("ckpt is dump at: ", ckpt_path_name)
