import numpy as np
import time
import os

from toolbox.interface.cross_agent import CrossAgentAnalyst
from collections import OrderedDict
from toolbox.interface.symbolic_agent_rollout import symbolic_agent_rollout

num_agents = 5
yaml_path = "data/yaml/0915-halfcheetah-ppo-20-agents.yaml"
num_rollouts = 10
num_children = 9
num_workers = 10

# normal_std = 0.1
normal_mean = 1.0

spawn_seed = 0
num_samples = 10 # From each agent's dataset
pca_dim = 50

dir_name = "notebooks/1012-scale-distance-relationship"

std_search_range = np.linspace(0.0, 1, 21)
std_ret_dict = {}


now = start = time.time()
std_load_obj_dict = OrderedDict()

for i, std in enumerate(std_search_range):
    print("[{}/{}] (+{:.2f}s/{:.2f}s) We are searching std: {}.".format(
        i + 1, len(std_search_range), time.time()-now, time.time()-start, std
    ))
    now = time.time()
    a, path = symbolic_agent_rollout(
        yaml_path, num_agents, num_rollouts,
        num_workers, num_children,
        std, normal_mean, dir_name
    )

    std_load_obj_dict[std] = a

    print("[{}/{}] (+{:.2f}s/{:.2f}s) Finshed std: {}! Save at: <{}>".format(
        i + 1, len(std_search_range), time.time()-now, time.time()-start, std,
        path
    ))
    std_ret_dict[std] = path

for std, path in std_ret_dict.items():
    print("Std: {}, Path: {}".format(std, path))
