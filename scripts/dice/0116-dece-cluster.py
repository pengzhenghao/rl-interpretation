# trainer.py
import os
import sys

import ray

redis_password = sys.argv[1]
num_cpus = int(sys.argv[2])

print("IP head we receive: ", os.environ["ip_head"])

ray.init(address=os.environ["ip_head"], redis_password=redis_password)

print(ray.available_resources())
print('*******************************************')
print('*******************************************')
print('*******************************************')
print('*******************************************')
print('*******************************************')
print("Nodes in the Ray cluster:")
print(ray.nodes())

from ray import tune
import pybullet_envs

print(pybullet_envs.getList())

from ray.tune.registry import register_env


def make_pybullet(_=None):
    import pybullet_envs
    import gym
    print("Successfully import pybullet and found: ", pybullet_envs.getList())
    return gym.make("Walker2DBulletEnv-v0")


register_env("Walker2DBulletEnv-v0", make_pybullet)

tune.run("PPO", config={'num_gpus': 1, 'env': "Walker2DBulletEnv-v0",
                        "seed": tune.grid_search(list(range(60)))},
         stop={"timesteps_total": 1000}, verbose=1)
# import os
# import sys
#
# from ray import tune
# import ray
#
# from toolbox.cooperative_exploration.train import train
# from toolbox.dece.dece import DECETrainer
# from toolbox.dece.utils import *
# from toolbox.marl import MultiAgentEnvWrapper
# import pybullet_envs
#
# redis_password = sys.argv[1]
# num_cpus = int(sys.argv[2])
#
# os.environ['OMP_NUM_THREADS'] = '1'
#
# # if __name__ == '__main__':
# redis_password = sys.argv[1]
# num_cpus = int(sys.argv[2])
#
# ray.init(address=os.environ["ip_head"], redis_password=redis_password)
#
#
# tune.run("PPO", config={'env': "Walker2DBulletEnv-v0"},
#          stop={"timesteps_total": 1000}, verbose=1)
#     # exp_name = "TEST0115-dece-cluster"
#     # env_name = "Walker2DBulletEnv-v0"
#     # num_seeds = 3
#     #
#     # assert os.getenv("OMP_NUM_THREADS") == '1'
#     #
#     # walker_config = {
#     #     DELAY_UPDATE: tune.grid_search([True, False]),
#     #     CONSTRAIN_NOVELTY: tune.grid_search(['hard', 'soft', None]),
#     #     REPLAY_VALUES: tune.grid_search([False]),
#     #     USE_DIVERSITY_VALUE_NETWORK: tune.grid_search([True, False]),
#     #
#     #     "env": MultiAgentEnvWrapper,
#     #     "env_config": {
#     #         "env_name": env_name,
#     #         "num_agents": tune.grid_search([5])
#     #     },
#     #
#     #     # should be fixed
#     #     "kl_coeff": 1.0,
#     #     "num_sgd_iter": 20,
#     #     "lr": 0.0002,
#     #     'sample_batch_size': 200,
#     #     'sgd_minibatch_size': 4000,
#     #     'train_batch_size': 60000,
#     #     "num_gpus": 0.5,
#     #     "num_cpus_per_worker": 1,
#     #     "num_cpus_for_driver": 1,
#     #     "num_envs_per_worker": 16,
#     #     'num_workers': 8,
#     # }
#     #
#     # train(
#     #     config=walker_config,
#     #     trainer=DECETrainer,
#     #     env_name=walker_config['env_config']['env_name'],
#     #     stop={"timesteps_total": int(5e7)},
#     #     exp_name=exp_name,
#     #     num_agents=walker_config['env_config']['num_agents'],
#     #     num_seeds=num_seeds,
#     #     num_gpus=None,
#     #     test_mode=False,
#     #     verbose=1,
#     #     address=os.environ['ip_head'],
#     #     redis_password=redis_password
#     # )
