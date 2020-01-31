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
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--stop", type=float, default=5e6)
    parser.add_argument("--env-name", type=str, default="BipedalWalker-v2")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    exp_name = args.exp_name

    mode = args.mode

    assert os.getenv("OMP_NUM_THREADS") == '1'

    test = args.test

    walker_config = {
        DELAY_UPDATE: tune.grid_search([True]),
        CONSTRAIN_NOVELTY: tune.grid_search(['soft']),
        REPLAY_VALUES: tune.grid_search([False]),
        USE_DIVERSITY_VALUE_NETWORK: tune.grid_search([False]),
        NORMALIZE_ADVANTAGE: tune.grid_search([False]),

        "env": MultiAgentEnvWrapper,
        "env_config": {
            "env_name": args.env_name,
            "num_agents": tune.grid_search([5])
        },

        # should be fixed
        "kl_coeff": 1.0,
        "entropy_coeff": 0.001,
        "lambda": 0.95,
        "num_sgd_iter": 10,
        "lr": 0.0003,
        'sample_batch_size': 50,
        'sgd_minibatch_size': 64,
        'train_batch_size': 2048,
        "num_gpus": 0.2,
        "num_cpus_per_worker": 1,
        "num_cpus_for_driver": 1,
        "num_envs_per_worker": 16,
        'num_workers': 1,
    }

    if mode == TWO_SIDE_CLIP_LOSS:
        walker_config[TWO_SIDE_CLIP_LOSS] = tune.grid_search([False])
    elif mode == NORMALIZE_ADVANTAGE:
        walker_config[NORMALIZE_ADVANTAGE] = tune.grid_search([True])
    elif mode == CONSTRAIN_NOVELTY:
        walker_config[CONSTRAIN_NOVELTY] = tune.grid_search(
            ['soft', 'hard', None])
    elif mode == DELAY_UPDATE:
        walker_config[DELAY_UPDATE] = tune.grid_search([False])
        # 3 trials
    elif mode == USE_DIVERSITY_VALUE_NETWORK:
        walker_config[USE_DIVERSITY_VALUE_NETWORK] = tune.grid_search(
            [True])
        # 3 trials
    elif mode == USE_BISECTOR:
        walker_config[USE_BISECTOR] = tune.grid_search([False])
        # 3 trials
    elif mode == DIVERSITY_REWARD_TYPE:
        walker_config[DIVERSITY_REWARD_TYPE] = tune.grid_search(['kl'])
        # 3 trials
    elif mode == ONLY_TNB:
        walker_config[ONLY_TNB] = tune.grid_search([True])
        # 3 trials

    train(
        extra_config=walker_config,
        trainer=DECETrainer,
        env_name=walker_config['env_config']['env_name'],
        stop={"timesteps_total": int(args.stop) if not test else 2000},
        exp_name=exp_name,
        num_agents=walker_config['env_config']['num_agents'],
        num_seeds=args.num_seeds,
        num_gpus=args.num_gpus,
        test_mode=args.test,
        verbose=1
    )
