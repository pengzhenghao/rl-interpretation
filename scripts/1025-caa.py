# from IPython.display import Image
import copy
import os
import os.path as osp
import pickle
import time
from collections import OrderedDict

import ray

from toolbox import initialize_ray
from toolbox.evaluate import MaskSymbolicAgent
from toolbox.evaluate.rollout import quick_rollout_from_symbolic_agents
from toolbox.interface.cross_agent import CrossAgentAnalyst

THIS_SCRIPT_IS_IN_TEST_MODE = False
num_agents = 10
num_rollouts = 10

num_workers = 16
dir_name = "./1023-cross-agent-retrain-NEW"
num_replay_workers = 16
os.makedirs(dir_name, exist_ok=True)

tt = time.time


def init_ray():
    initialize_ray(num_gpus=4, test_mode=True, local_mode=False,
                   object_store_memory=40 * int(1e9))


def shut_ray():
    ray.shutdown()


init_ray()


def remote_restore_and_compute(rollout_ret, now, start, dir_name, std):
    cluster_dataframe_inside = []

    agent_rollout_dict = OrderedDict()
    name_agent_info_mapping = OrderedDict()
    for key, (rd, ag) in rollout_ret.items():
        agent_rollout_dict[key] = rd
        name_agent_info_mapping[key] = ag

    print("[STD={}] (+{:.2f}s/{:.2f}s) Prepared to create CAA".format(
        std, tt() - now, tt() - start)
    )
    now = tt()
    caa = CrossAgentAnalyst()
    caa.feed(agent_rollout_dict, name_agent_info_mapping,
             num_replay_workers=num_replay_workers)

    print("[STD={}] (+{:.2f}s/{:.2f}s) Start to walkthrough CAA".format(
        std, tt() - now, tt() - start)
    )
    now = tt()
    result = caa.walkthrough()

    print(
        "[STD={}] (+{:.2f}s/{:.2f}s) Start to collect representation "
        "CAA".format(
            std, tt() - now, tt() - start))
    now = tt()
    caa.cluster_representation()

    print("[STD={}] (+{:.2f}s/{:.2f}s) Start to cal distances CAA".format(
        std, tt() - now, tt() - start))
    now = tt()
    caa.cluster_distance()

    print(
        "[STD={}] (+{:.2f}s/{:.2f}s) Start to collect summary CAA".format(
            std, tt() - now, tt() - start))
    now = tt()
    smr = copy.deepcopy(caa.summary())

    print(
        "[STD={}] (+{:.2f}s/{:.2f}s) Start to collect clutser_dataframe "
        "CAA".format(
            std, tt() - now, tt() - start))
    now = tt()
    precision_dict = caa.cluster_representation_precision_dict
    parent_cluster_dict = caa.cluster_representation_parent_cluster_dict

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


start = now = time.time()

with open("1023-cross-agent-retrain/retrain_agent_result_std=0.9-copy.pkl",
          'rb') as f:
    data = pickle.load(f)

ckpt = {
    "path": None,
    "run_name": "PPO",
    "env_name": "BipedalWalker-v2",
    "name": "test agent"
}

nest_agent = OrderedDict()
for std, agent_dict in data.items():
    nest_agent[std] = OrderedDict()
    for name, (_, weights) in agent_dict.items():
        nest_agent[std][name] = MaskSymbolicAgent(
            ckpt, existing_weights=weights
        )

print("Finish prepare symbolic agents.")

std_ret_rollout_dict_new = OrderedDict()
for std, agent_dict in nest_agent.items():
    print("Enter STD={}, quick rollout start!".format(std))
    rollout_ret = quick_rollout_from_symbolic_agents(
        agent_dict, num_rollouts, num_workers,
        env_wrapper=None  # This is not mujoco env!!
    )
    std_ret_rollout_dict_new[std] = rollout_ret

print("Finish rollout")
start = now = time.time()
std_summary_dict_new = OrderedDict()
cluster_dataframe_new = []
joint_cluster_df_dict_new = OrderedDict()
joint_prediction_dict_dict_new = OrderedDict()

os.makedirs(osp.join(dir_name, "caa_result_new"), exist_ok=True)
last_ckpt_path_name = None

for i, (std, rollout_ret) in enumerate(std_ret_rollout_dict_new.items()):

    ckpt_path_name = osp.join(dir_name,
                              "caa_result_new/CAA_result_std={}.pkl".format(
                                  std))

    shut_ray()
    init_ray()
    cluster_representation_cluster_df_dict, \
    cluster_representation_prediction_dict, \
    cluster_dataframe_single, \
    summary = remote_restore_and_compute(
        rollout_ret, now, start, dir_name, std
    )

    joint_cluster_df_dict_new[std] = cluster_representation_cluster_df_dict
    joint_prediction_dict_dict_new[
        std] = cluster_representation_prediction_dict
    std_summary_dict_new[std] = summary
    cluster_dataframe_new.append(cluster_dataframe_single)

    print("[{}/{}] (+{:.2f}s/{:.2f}s) Finshed std: {}!".format(
        i + 1, len(std_ret_rollout_dict_new), time.time() - now,
        time.time() - start,
        std
    ))
    now = time.time()

    # Used for checkpoint.
    if last_ckpt_path_name is not None and os.path.exists(last_ckpt_path_name):
        os.remove(last_ckpt_path_name)
    with open(ckpt_path_name, 'wb') as f:
        pickle.dump(
            [joint_cluster_df_dict_new, joint_prediction_dict_dict_new,
             std_summary_dict_new, cluster_dataframe_new], f
        )
        print("ckpt is dump at: ", ckpt_path_name)
    last_ckpt_path_name = ckpt_path_name

ckpt_path_name = osp.join(dir_name, "caa_result_new/CAA_result_final.pkl")
with open(ckpt_path_name, 'wb') as f:
    pickle.dump(
        [joint_cluster_df_dict_new, joint_prediction_dict_dict_new,
         std_summary_dict_new, cluster_dataframe_new], f
    )
    print("ckpt is dump at: ", ckpt_path_name)
