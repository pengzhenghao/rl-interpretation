from ray import tune

from toolbox.cooperative_exploration.train import train
from toolbox.marl import MultiAgentEnvWrapper
from toolbox.dece.utils import *
from toolbox.dece.dece import DECETrainer

if __name__ == '__main__':
    exp_name = "1231-dece-first-test"

    walker_config = {
        DELAY_UPDATE: tune.grid_search([True, False]),
        REPLAY_VALUES: tune.grid_search([True, False]),
        "env": MultiAgentEnvWrapper,
        "env_config": {
            "env_name": "Walker2d-v3",
            "num_agents": 5
        },

        # should be fixed
        "kl_coeff": 1.0,
        "num_sgd_iter": 20,
        "lr": 0.0001,
        'sample_batch_size': 256,
        'sgd_minibatch_size': 4096,
        'train_batch_size': 65536,
        "num_gpus": 0.45,
        "num_cpus_per_worker": 0.5,
        "num_cpus_for_driver": 0.5,
        "num_envs_per_worker": 16,
        'num_workers': 8,
    }
    train(
        extra_config=walker_config,
        trainer=DECETrainer,
        env_name=walker_config['env_config']['env_name'],
        stop={"timesteps_total": int(5e7)},
        exp_name=exp_name,
        num_agents=walker_config['env_config']['num_agents'],
        num_seeds=3,
        num_gpus=8,
        test_mode=False
    )
