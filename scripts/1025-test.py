from collections import OrderedDict

from toolbox import initialize_ray
from toolbox.evaluate import MaskSymbolicAgent
from toolbox.evaluate.rollout import quick_rollout_from_symbolic_agents

num_agents = 100
num_rollouts = 1
num_workers = 16



initialize_ray(num_gpus=4, test_mode=True)
ckpt = {
    "run_name": "PPO",
    "env_name": "BipedalWalker-v2",
    "path": None
}
agent_dict = OrderedDict()
for i in range(num_agents):
    ckpt['name'] = i
    agent_dict[i] = MaskSymbolicAgent(ckpt
                                      )

std_ret_rollout_dict_new = OrderedDict()
rollout_ret = quick_rollout_from_symbolic_agents(
    agent_dict, num_rollouts, num_workers,
    env_wrapper=None  # This is not mujoco env!!
)
