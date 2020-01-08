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
    parser.add_argument("--exp-name", type=str, default="0109-dece-vtrace")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    exp_name = args.exp_name

    assert os.getenv("OMP_NUM_THREADS") == '1'

    test = args.test

    walker_config = {
        # "use_vtrace": True,

        DELAY_UPDATE: tune.grid_search([True, False]),
        REPLAY_VALUES: tune.grid_search([False, False]),
        # USE_DIVERSITY_VALUE_NETWORK: tune.grid_search([True, False]),

        "env": MultiAgentEnvWrapper,
        "env_config": {
            "env_name": "Walker2d-v3",
            "num_agents": tune.grid_search([5, 1])
        },

        # should be fixed
        "kl_coeff": 1.0,
        "num_sgd_iter": 20,
        "lr": 0.0001,
        'sample_batch_size': 256 if not test else 32,
        'sgd_minibatch_size': 4096 if not test else 32,
        'train_batch_size': 65536 if not test else 256,
        "num_gpus": 0.45,
        "num_cpus_per_worker": 0.8,
        "num_cpus_for_driver": 0.5,
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
        test_mode=args.test
    )
