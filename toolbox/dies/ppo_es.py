from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.tune.utils import merge_dicts

from toolbox import initialize_ray, train
from toolbox.dies.es_utils import run_evolution_strategies
from toolbox.marl import get_marl_env_config, on_train_result

ppo_es_default_config = merge_dicts(
    DEFAULT_CONFIG,
    dict(update_steps=100000, callbacks={"on_train_result": on_train_result})
)

PPOESTrainer = PPOTrainer.with_updates(
    name="PPOES",
    default_config=ppo_es_default_config,
    after_train_result=run_evolution_strategies
)

if __name__ == '__main__':
    env_name = "CartPole-v0"
    num_agents = 3
    config = {
        "num_sgd_iter": 2,
        "train_batch_size": 400,
        "update_steps": 1000,
        **get_marl_env_config(env_name, num_agents)
    }
    initialize_ray(test_mode=True, local_mode=True)
    train(
        PPOESTrainer,
        config,
        exp_name="DELETE_ME_TEST",
        stop={"timesteps_total": 10000},
        test_mode=True
    )
