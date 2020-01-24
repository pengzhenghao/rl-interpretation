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
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--exp-name", type=str,
                        default="0116-dece")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--num-seeds", type=int, default=3)
    # parser.add_argument("--address", type=str, required=True)
    parser.add_argument("--env-name", type=str, default="Walker2d-v3")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    mode = args.mode
    exp_name = args.exp_name + "-{}".format(mode)

    assert os.getenv("OMP_NUM_THREADS") == '1'
    assert args.env_name in ['Walker2d-v3', 'Hopper-v3', 'HalfCheetah-v3']

    test = args.test

    walker_config = {
        REPLAY_VALUES: tune.grid_search([False]),

        DELAY_UPDATE: tune.grid_search([True]),
        CONSTRAIN_NOVELTY: tune.grid_search(['soft']),
        USE_DIVERSITY_VALUE_NETWORK: tune.grid_search([True]),
        USE_BISECTOR: tune.grid_search([True]),
        DIVERSITY_REWARD_TYPE: tune.grid_search(['mse']),
        ONLY_TNB: tune.grid_search([False]),

        "env": MultiAgentEnvWrapper,
        "env_config": {
            "env_name": args.env_name,
            "num_agents": tune.grid_search([5])  # quick run 5 agent firstly
        },

        # should be fixed
        "kl_coeff": 1.0,
        "num_sgd_iter": 20,
        "lr": 0.0002,
        'sample_batch_size': 200 if not test else 40,
        'sgd_minibatch_size': 1000 if not test else 200,
        'train_batch_size': 10000 if not test else 400,
        "num_gpus": 0.45,
        "num_cpus_per_worker": 0.45,
        "num_cpus_for_driver": 0.4,
        "num_envs_per_worker": 8 if not test else 1,
        'num_workers': 8 if not test else 1,
    }

    assert mode in [
        TWO_SIDE_CLIP_LOSS, PURE_OFF_POLICY, NORMALIZE_ADVANTAGE
    ]

    if mode == TWO_SIDE_CLIP_LOSS:
        walker_config[TWO_SIDE_CLIP_LOSS] = tune.grid_search([False])
    elif mode == PURE_OFF_POLICY:
        walker_config[PURE_OFF_POLICY] = tune.grid_search([True])
    elif mode == NORMALIZE_ADVANTAGE:
        walker_config[NORMALIZE_ADVANTAGE] = tune.grid_search([False])

    train(
        extra_config=walker_config,
        trainer=DECETrainer,
        env_name=walker_config['env_config']['env_name'],
        stop={"timesteps_total": int(5e7) if not test else 2000},
        exp_name=exp_name,
        num_agents=walker_config['env_config']['num_agents'],
        num_seeds=args.num_seeds,
        num_gpus=None,
        test_mode=args.test,
        # address=args.address,
        verbose=1
    )
