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
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--env-name", type=str, default="Walker2d-v3")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    exp_name = args.exp_name

    assert os.getenv("OMP_NUM_THREADS") == '1'

    test = args.test

    walker_config = {
        DELAY_UPDATE: tune.grid_search([True]),
        # CONSTRAIN_NOVELTY: tune.grid_search(['soft', 'hard', None]),
        REPLAY_VALUES: tune.grid_search([False]),
        # USE_DIVERSITY_VALUE_NETWORK: tune.grid_search([False]),
        # NORMALIZE_ADVANTAGE: tune.grid_search([False]),
        USE_BISECTOR: tune.grid_search([False]),

        "env": MultiAgentEnvWrapper,
        "env_config": {
            "env_name": args.env_name,
            "num_agents": tune.grid_search([5])
        },

        # should be fixed
        "kl_coeff": 1.0,
        "num_sgd_iter": 10,
        "lr": 0.0002,
        'sample_batch_size': 50,
        'sgd_minibatch_size': 64,
        'train_batch_size': 2048,
        "num_gpus": 0.4,
        "num_cpus_per_worker": 1,
        "num_cpus_for_driver": 1,
        "num_envs_per_worker": 5,
        'num_workers': 1,
    }
    train(
        extra_config=walker_config,
        trainer=DECETrainer,
        env_name=walker_config['env_config']['env_name'],
        stop={"timesteps_total": int(5e6) if not test else 2000},
        exp_name=exp_name,
        num_agents=walker_config['env_config']['num_agents'],
        num_seeds=args.num_seeds,
        num_gpus=args.num_gpus,
        test_mode=args.test,
        verbose=1
    )
