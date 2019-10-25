from collections import OrderedDict
import numpy as np
from toolbox import initialize_ray
from toolbox.evaluate import MaskSymbolicAgent
from toolbox.evaluate.rollout import quick_rollout_from_symbolic_agents
from toolbox.evaluate.replay import RemoteSymbolicReplayManager

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

# std_ret_rollout_dict_new = OrderedDict()
# rollout_ret = quick_rollout_from_symbolic_agents(
#     agent_dict, num_rollouts, num_workers,
#     env_wrapper=None  # This is not mujoco env!!
# )
# print(rollout_ret)
# print(np.mean(rollout_ret), np.min(rollout_ret), np.max(rollout_ret))

replay_manager = RemoteSymbolicReplayManager(
    num_workers, total_num=len(agent_dict)
)

for i, (name, symbolic_agent) in \
        enumerate(agent_dict.items()):
    replay_manager.replay(name, symbolic_agent, np.ones((1000, 24)))

print("[INSIDE CAA][replay] have submitted the all commands to RSRM")
replay_result = replay_manager.get_result()
print("[INSIDE CAA][replay] have ge_result() from RSRM")