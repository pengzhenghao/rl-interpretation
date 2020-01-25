import copy
import pickle

from ray import tune

from toolbox import initialize_ray
from toolbox.dece.dece import DECETrainer
from toolbox.marl import MultiAgentEnvWrapper
from toolbox.utils import get_local_dir

ppo_default_config = {
    "seed": tune.grid_search([i * 100 for i in range(3)]),
    "env": "DeceptiveMaze-v0",

    "num_sgd_iter": 10,
    "lr": 0.001,
    'sample_batch_size': 16,
    'sgd_minibatch_size': 32,
    'train_batch_size': 512,
    "num_gpus": 0.2,
    "num_envs_per_worker": 4,
    'num_workers': 1,
    "model": {"fcnet_hiddens": [64, 64]}
}

initialize_ray(
    test_mode=False,
    local_mode=False,
    num_gpus=4)

num_agents = 4

dece_config = copy.deepcopy(ppo_default_config)
dece_config.update(
    {
        "env": MultiAgentEnvWrapper,
        "constrain_novelty": tune.grid_search(['hard', 'soft', None]),
        "env_config": {
            "env_name": ppo_default_config['env'],
            "num_agents": num_agents
        },
        "alpha_coefficient": tune.grid_search([0.05, 0.1, 0.01]),
        "tau": tune.grid_search([0.05, 0.1, 0.01]),
        "novelty_stat_length": tune.grid_search([20, 50])
    },
)

dece_analysis = tune.run(
    DECETrainer,
    local_dir=get_local_dir(),
    name="0126-dece-DeceptiveMaze",
    verbose=1,
    stop={"info/num_steps_sampled": 200000},
    config=dece_config,

    checkpoint_freq=10,
    checkpoint_at_end=True
)

path = "{}-{}-{}ts.pkl".format(
    "0126-dece-DeceptiveMaze", "DeceptiveMaze-v0", 200000
)

with open(path, "wb") as f:
    data = dece_analysis.fetch_trial_dataframes()
    pickle.dump(data, f)
    print("Result is saved at: <{}>".format(path))

initialize_ray(
    test_mode=False,
    local_mode=False,
    num_gpus=4)

ppo_analysis = tune.run(
    "PPO",
    local_dir=get_local_dir(),
    name="0126-ppo-DeceptiveMaze",
    verbose=1,
    stop={"info/num_steps_sampled": 200000},
    config=ppo_default_config,
    checkpoint_freq=10,
    checkpoint_at_end=True
)

path = "{}-{}-{}ts.pkl".format(
    "0126-ppo-DeceptiveMaze", "DeceptiveMaze-v0", 200000
)

with open(path, "wb") as f:
    data = ppo_analysis.fetch_trial_dataframes()
    pickle.dump(data, f)
    print("Result is saved at: <{}>".format(path))
