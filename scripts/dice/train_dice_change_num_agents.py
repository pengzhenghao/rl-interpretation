import argparse
import os

from ray import tune

from toolbox.marl import MultiAgentEnvWrapper
from .train_dice_change_env import train

os.environ['OMP_NUM_THREADS'] = '1'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--num-agents", type=int, default=5)
    parser.add_argument("--env-name", type=str, default="Walker2d-v3")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    env_name = args.env_name
    num_agents = int(args.num_agents)
    exp_name = "{}-{}-{}agents".format(args.exp_name, env_name, num_agents)
    stop = int(5e7)
    large = True

    config = {
        "env": MultiAgentEnvWrapper,
        "env_config": {
            "env_name": env_name,
            "num_agents": tune.grid_search([num_agents])
        },
        "kl_coeff": 1.0,
        "num_sgd_iter": 10,
        "lr": 0.0001,
        'sample_batch_size': 200 if large else 50,
        'sgd_minibatch_size': 100 if large else 64,
        'train_batch_size': 10000 if large else 2048,
        "num_gpus": 0.4,
        "num_cpus_per_worker": 0.4,
        "num_cpus_for_driver": 0.45,
        "num_envs_per_worker": 8 if large else 5,
        'num_workers': 8 if large else 1,
    }

    train(
        extra_config=config,
        env_name=config['env_config']['env_name'],
        stop=stop,
        exp_name=exp_name,
        num_agents=config['env_config']['num_agents'],
        num_seeds=args.num_seeds,
        num_gpus=args.num_gpus,
        test_mode=args.test,
        verbose=1
    )
