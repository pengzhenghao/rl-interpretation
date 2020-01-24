import argparse
import os

from ray import tune

from toolbox.cooperative_exploration.train import train
from toolbox.dece.dece import DECETrainer
from toolbox.dece.utils import *
from toolbox.marl import MultiAgentEnvWrapper

os.environ['OMP_NUM_THREADS'] = '1'
GB = 1024 * 1024 * 1024

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str,
                        default="0122-dece-humanoid")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--env-name", type=str, default="Walker2d-v3")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    exp_name = args.exp_name

    assert os.getenv("OMP_NUM_THREADS") == '1'

    assert args.env_name in [
        'Walker2DBulletEnv-v0',
        'HalfCheetahBulletEnv-v0',
        'AntBulletEnv-v0',
        'HopperBulletEnv-v0',
        'HumanoidBulletEnv-v0',
        'HumanoidFlagrunBulletEnv-v0'
    ]

    test = args.test

    walker_config = {
        DELAY_UPDATE: tune.grid_search([True]),
        CONSTRAIN_NOVELTY: tune.grid_search(['soft']),
        REPLAY_VALUES: tune.grid_search([False]),
        USE_DIVERSITY_VALUE_NETWORK: tune.grid_search([True]),

        "env": MultiAgentEnvWrapper,
        "env_config": {
            "env_name": args.env_name,
            "num_agents": tune.grid_search([5])
        },

        # should be fixed
        "kl_coeff": 1.0,
        "gamma": 0.995,
        "lambda": 0.95,
        "clip_param": 0.2,
        "num_sgd_iter": 20,
        "lr": 0.0001,
        "horizon": 5000,

        'sample_batch_size': 200,
        'sgd_minibatch_size': 10000,
        'train_batch_size': 100000,
        "num_gpus": 1,
        "num_cpus_per_worker": 1,
        "num_cpus_for_driver": 1,
        "num_envs_per_worker": 8,
        'num_workers': 24,

        "object_store_memory": int(5 * GB),
        "memory": int(25 * GB)
    }

    train(
        extra_config=walker_config,
        trainer=DECETrainer,
        env_name=walker_config['env_config']['env_name'],
        stop={"timesteps_total": int(5e8)},
        exp_name=exp_name,
        num_agents=walker_config['env_config']['num_agents'],
        num_seeds=args.num_seeds,
        num_gpus=args.num_gpus,
        test_mode=args.test,
        verbose=1,
        init_memory=int(70 * GB),
        init_object_store_memory=int(15 * GB),
        init_redis_max_memory=int(6 * GB),
    )
