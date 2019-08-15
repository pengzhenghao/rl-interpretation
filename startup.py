import ray
from ray import tune
import logging

# The arguments below is copied from
# https://github.com/araffin/rl-baselines-zoo/blob/master/hyperparams/ppo2.yml

# Notes for BipedalWalker: BipedalWalker-v2 defines "solving" as getting
# average reward of 300 over 100 consecutive trials

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--exp-name", type=str, required=True)
args = parser.parse_args()

ray.init(logging_level=logging.ERROR, log_to_driver=False)
tune.run(
    "A2C",
    name=args.exp_name,
    verbose=1,
    checkpoint_freq=100,
    checkpoint_at_end=True,
    stop={"timesteps_total": int(5e6)},
    # stop={"episode_reward_mean": 20},
    config={
        # "env": "BreakoutNoFrameskip-v4",
        "env": "BipedalWalker-v2",
        "seed": tune.grid_search(list(range(300))),
        # "lambda": 0.95,
        "entropy_coeff": 0.0,
        # "clip_param": 0.2,
        "num_workers": 1,
        "num_envs_per_worker": 16,
        # "num_sgd_iter": 10,
        # "lr": 2.5e-4,
        "num_gpus": 0.15,
        "num_cpus_for_driver": 0.2
    },
)
