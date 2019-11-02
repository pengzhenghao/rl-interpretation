# from IPython.display import Image
import copy
import os
import os.path as osp
import pickle
import time
from collections import OrderedDict

import numpy as np
import ray

from toolbox import initialize_ray
# from toolbox.evaluate.rollout import rollout
from toolbox.env.mujoco_wrapper import MujocoWrapper
from toolbox.evaluate.rollout import quick_rollout_from_symbolic_agents
from toolbox.interface.cross_agent import CrossAgentAnalyst
from toolbox.interface.symbolic_agent_rollout import symbolic_agent_rollout
from toolbox.train.finetune import RemoteSymbolicTrainManager

THIS_SCRIPT_IS_IN_TEST_MODE = False

if THIS_SCRIPT_IS_IN_TEST_MODE:
    """**************************************************
    *****************************************************
    ******************** TEST HYPERPARAMETERS!!! ********
    **************************************************"""
    num_agents = 10
    yaml_path = "../data/yaml/ppo-300-agents.yaml"
    num_rollouts = 2
    num_children = 9
    num_workers = 16
    normal_mean = 1.0
    #     num_train_iters = 5
    dir_name = "./DELETEME_DATASET_1023"
    num_replay_workers = 16

    stop_criterion = {
        "episode_reward_mean": -100
    }

    std_search_range = [0.25, 0.5]

    pca_dim = 50
    fig_dir_name = osp.join(dir_name, 'fig')
    os.makedirs(dir_name, exist_ok=True)
    os.makedirs(fig_dir_name, exist_ok=True)

else:
    num_agents = 10
    yaml_path = "../data/yaml/ppo-300-agents.yaml"
    num_rollouts = 10
    num_children = 9
    num_workers = 16
    normal_mean = 1.0
    dir_name = "./1023-cross-agent-retrain"
    num_replay_workers = 16

    stop_criterion = {
        "episode_reward_mean": 280,
        "timesteps_since_restore": 500000,
        #         "time_since_restore": 3600,
    }

    std_search_range = np.linspace(0, 1, 21)

    pca_dim = 50
    fig_dir_name = osp.join(dir_name, 'fig')
    os.makedirs(dir_name, exist_ok=True)
    os.makedirs(fig_dir_name, exist_ok=True)


def init_ray():
    initialize_ray(num_gpus=4, test_mode=THIS_SCRIPT_IS_IN_TEST_MODE,
                   object_store_memory=40 * int(1e9))


def shut_ray():
    ray.shutdown()


std_ret_rollout_dict = {}

start = now = time.time()

for i, std in enumerate(std_search_range):
    shut_ray()
    init_ray()

    rollout_ret, _ = symbolic_agent_rollout(
        yaml_path, num_agents, num_rollouts,
        num_workers, num_children,
        std, normal_mean, dir_name,
        # store=False, clear_at_end=False
        # We don't store it. We just need the result.
    )

    std_ret_rollout_dict[std] = rollout_ret

ckpt_path_name = osp.join(dir_name, "std_ret_rollout_dict.pkl")
with open(ckpt_path_name, 'wb') as f:
    pickle.dump(
        std_ret_rollout_dict, f
    )
    print("ckpt is dump at: ", ckpt_path_name)


"""
std_agent_dict_dict looks like this:

OrderedDict([(1.0,
              OrderedDict([('PPO seed=102 rew=271.39 child=0',
                            <toolbox.evaluate.symbolic_agent.MaskSymbolicAgent at 0x7fe1d77f92b0>),
                           ('PPO seed=102 rew=271.39 child=1',
                            <toolbox.evaluate.symbolic_agent.MaskSymbolicAgent at 0x7fe1d77f9eb8>),
                           ('PPO seed=102 rew=271.39 child=2',
                            <toolbox.evaluate.symbolic_agent.MaskSymbolicAgent at 0x7fe1d77f9908>),
                           ('PPO seed=121 rew=299.35 child=0',
                            <toolbox.evaluate.symbolic_agent.MaskSymbolicAgent at 0x7fe1d77f9e10>),
                           ('PPO seed=121 rew=299.35 child=1',
                            <toolbox.evaluate.symbolic_agent.MaskSymbolicAgent at 0x7fe1d52a5048>),
                           ('PPO seed=121 rew=299.35 child=2',
                            <toolbox.evaluate.symbolic_agent.MaskSymbolicAgent at 0x7fe1d52a5080>)]))])
"""

std_agent_dict_dict = OrderedDict()
for std, content in std_ret_rollout_dict.items():
    agent_dict = OrderedDict()
    for name, (_, agent) in content.items():
        agent_dict[name] = agent
    std_agent_dict_dict[std] = agent_dict

ckpt_path_name = osp.join(dir_name, "std_agent_dict_dict.pkl")
with open(ckpt_path_name, 'wb') as f:
    pickle.dump(
        std_agent_dict_dict, f
    )
    print("ckpt is dump at: ", ckpt_path_name)


tt = time.time


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

std_summary_dict = OrderedDict()
cluster_dataframe = []
joint_cluster_df_dict = OrderedDict()
joint_prediction_dict_dict = OrderedDict()

os.makedirs(osp.join(dir_name, "caa_result"), exist_ok=True)
last_ckpt_path_name = None

for i, (std, rollout_ret) in enumerate(std_ret_rollout_dict.items()):

    ckpt_path_name = osp.join(dir_name,
                              "caa_result/CAA_result_std={}.pkl".format(std))

    shut_ray()
    init_ray()

    cluster_representation_cluster_df_dict, \
    cluster_representation_prediction_dict, \
    cluster_dataframe_single, \
    summary = remote_restore_and_compute(
        rollout_ret, now, start, dir_name, std
    )

    joint_cluster_df_dict[std] = cluster_representation_cluster_df_dict
    joint_prediction_dict_dict[
        std] = cluster_representation_prediction_dict
    std_summary_dict[std] = summary
    cluster_dataframe.append(cluster_dataframe_single)

    print("[{}/{}] (+{:.2f}s/{:.2f}s) Finshed std: {}!".format(
        i + 1, len(std_ret_rollout_dict), time.time() - now,
        time.time() - start,
        std
    ))
    now = time.time()

    # Used for checkpoint.
    if last_ckpt_path_name is not None and os.path.exists(last_ckpt_path_name):
        os.remove(last_ckpt_path_name)
    with open(ckpt_path_name, 'wb') as f:
        pickle.dump(
            [joint_cluster_df_dict, joint_prediction_dict_dict,
             std_summary_dict, cluster_dataframe], f
        )
        print("ckpt is dump at: ", ckpt_path_name)
    last_ckpt_path_name = ckpt_path_name

ckpt_path_name = osp.join(dir_name, "caa_result/CAA_result_final.pkl")
with open(ckpt_path_name, 'wb') as f:
    pickle.dump(
        [joint_cluster_df_dict, joint_prediction_dict_dict,
         std_summary_dict,
         cluster_dataframe], f
    )
    print("ckpt is dump at: ", ckpt_path_name)

initial_states_of_agents = [joint_cluster_df_dict, joint_prediction_dict_dict,
                            std_summary_dict,
                            cluster_dataframe]

initial_states_of_agents = copy.deepcopy(initial_states_of_agents)

std_retrain_result_dict = OrderedDict()
last_ckpt = None

for i, (std, agent_dict) in enumerate(std_agent_dict_dict.items()):
    shut_ray()
    init_ray()
    train_manager = RemoteSymbolicTrainManager(num_workers, len(agent_dict))
    for name, agent in agent_dict.items():
        train_manager.train(name, agent, stop_criterion)

    result = train_manager.get_result()
    std_retrain_result_dict[std] = copy.deepcopy(result)

    if last_ckpt is not None and os.path.exists(last_ckpt):
        os.remove(last_ckpt)

    ckpt_path_name = osp.join(dir_name,
                              "retrain_agent_result_std={}.pkl".format(std))
    with open(ckpt_path_name, 'wb') as f:
        pickle.dump(
            std_retrain_result_dict, f
        )
        print("ckpt is dump at: ", ckpt_path_name)
    last_ckpt = ckpt_path_name

ckpt_path_name = osp.join(dir_name, "retrain_agent_result_final.pkl")
with open(ckpt_path_name, 'wb') as f:
    pickle.dump(
        std_retrain_result_dict, f
    )
    print("ckpt is dump at: ", ckpt_path_name)

std_ret_rollout_dict_new = OrderedDict()

for std, agent_dict in std_agent_dict_dict.items():
    rollout_ret = quick_rollout_from_symbolic_agents(
        agent_dict, num_rollouts, num_workers, MujocoWrapper
    )
    std_ret_rollout_dict_new[std] = rollout_ret

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
        i + 1, len(std_ret_rollout_dict), time.time() - now,
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
