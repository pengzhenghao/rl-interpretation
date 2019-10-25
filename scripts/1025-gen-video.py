import os.path as osp
import pickle
from collections import OrderedDict

import ray

from toolbox.evaluate import MaskSymbolicAgent
from toolbox.utils import initialize_ray
from toolbox.visualize.generate_trailer import RemoteSymbolicAgentVideoManager


def ir():
    initialize_ray(test_mode=False, num_gpus=2)


def sr():
    ray.shutdown()


with open("retrain_agent_result_std=0.9-copy.pkl", 'rb') as f:
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
        nest_agent[std][name] = MaskSymbolicAgent(ckpt,
                                                  existing_weights=weights)

num_workers = 4
base_output_path = "./videos"
for std, agent_dict in nest_agent.items():

    sr()
    ir()

    num_agents = len(nest_agent[std])

    rsavm = RemoteSymbolicAgentVideoManager(num_workers, num_agents)

    std_base_output_path = osp.join(base_output_path, "std={:.3f}".format(std))

    for i, (name, symbolic_agent) in enumerate(nest_agent[std].items()):
        rsavm.generate_video(name, symbolic_agent, std_base_output_path)

    result = rsavm.get_result()
    print(result)
