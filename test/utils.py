import os

from evaluate.rollout import RolloutWorkerWrapper
from utils import build_env


def get_test_agent_config():
    return {
        "env_maker": build_env,
        "ckpt": os.path.abspath(os.path.expanduser(
            "~/ray_results/0810-20seeds/"
            "PPO_BipedalWalker-v2_0_seed=0_2019-08-10_15-21-164grca382/"
            "checkpoint_313/checkpoint-313")),
        "env_name": "BipedalWalker-v2",
        "run_name": "PPO"
    }


def load_test_agent_rollout():
    config = get_test_agent_config()
    num_rollout = 2
    seed = 0
    rww_new = RolloutWorkerWrapper()
    rww_new.reset(
        config['ckpt'], num_rollout, seed, config['env_maker'],
        config['run_name'], config['env_name']
    )
    return rww_new.wrap_sample()
