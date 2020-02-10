import argparse
import os

from ray import tune

from toolbox.cooperative_exploration.train import train
from toolbox.dece.dece import DECETrainer
from toolbox.dece.utils import *
from toolbox.marl import MultiAgentEnvWrapper

os.environ['OMP_NUM_THREADS'] = '1'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--num-agents", type=int, default=5)
    parser.add_argument("--env-name", type=str, default="Walker2d-v3")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--address", type=str, required=True)
    args = parser.parse_args()
    exp_name = args.exp_name

    assert os.getenv("OMP_NUM_THREADS") == '1'

    test = args.test

    walker_config = {
        DELAY_UPDATE: tune.grid_search([True]),
        CONSTRAIN_NOVELTY: tune.grid_search(["soft"]),
        REPLAY_VALUES: tune.grid_search([False]),
        USE_DIVERSITY_VALUE_NETWORK: tune.grid_search([True]),

        "env": MultiAgentEnvWrapper,
        "env_config": {
            "env_name": args.env_name,
            "num_agents": tune.grid_search([args.num_agents])
        },

        # should be fixed
        "kl_coeff": 1.0,
        "num_sgd_iter": 20,
        "lr": 0.0002,
        'sample_batch_size': 200 if not test else 40,
        'sgd_minibatch_size': 1000 if not test else 200,
        'train_batch_size': 10000 if not test else 400,
        "num_gpus": 0.5,
        "num_cpus_per_worker": 0.8,
        "num_cpus_for_driver": 0.8,
        "num_envs_per_worker": 8 if not test else 1,
        'num_workers': 8 if not test else 1,
    }
    train(
        extra_config=walker_config,
        trainer=DECETrainer,
        env_name=walker_config['env_config']['env_name'],
        stop={"timesteps_total": int(5e7) if not test else 2000},
        exp_name=exp_name,
        num_agents=walker_config['env_config']['num_agents'],
        num_seeds=args.num_seeds,
        num_gpus=args.num_gpus,
        test_mode=args.test,
        address=args.address,
        verbose=1
    )
