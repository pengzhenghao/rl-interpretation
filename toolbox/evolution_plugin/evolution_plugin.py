import copy

import ray
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG, \
    validate_config as original_validate
from ray.tune.utils import merge_dicts

from toolbox import initialize_ray, train
from toolbox.evolution import GaussianESTrainer
from toolbox.evolution.modified_es import DEFAULT_CONFIG as es_config

# TODO support ARS default config

ESPlugin = GaussianESTrainer

# ARSPlugin = GaussianARSTrainer


ppo_es_default_config = merge_dicts(
    DEFAULT_CONFIG,
    dict(evolution=es_config)
)


def validate_config(config):
    original_validate(config)
    config["model"]["vf_share_layers"] = config["vf_share_layers"]
    config["evolution"]["model"] = copy.deepcopy(config["model"])
    assert not config["evolution"]["model"]["vf_share_layers"]


def after_optimizer_step(trainer, feteches):
    print("sss")
    pass


def after_init(trainer):
    """
    TODO: Launch the first ES task here
    """
    base = ESPlugin

    @ray.remote(num_gpus=0)
    class EvolutionPluginRemote(base):
        def sync_weights(self, weights):
            self.get_policy().variables.set_weights(weights)

        def retrieve_weights(self):
            return self.get_policy().variables.get_weights()

        def step(self):
            train_result = self.train()
            weights = self.get_weights()  # flatten weights
            return train_result, weights

    trainer._evolution_plugin = EvolutionPluginRemote.remote(
        trainer.config["evolution"], trainer.config["env"])

    _sync_weights(trainer, trainer._evolution_plugin)


def _sync_weights(trainer, plugin):
    # FIXME never sync the weights of value network.
    plugin.sync_weights.remote(
        trainer.get_policy().get_weights()
    )


EPTrainer = PPOTrainer.with_updates(
    name="EvolutionPlugin",
    default_config=ppo_es_default_config,
    after_init=after_init,
    after_optimizer_step=after_optimizer_step,
    # after_train_result=run_evolution_strategies,
    validate_config=validate_config
)

if __name__ == '__main__':
    env_name = "CartPole-v0"
    # num_agents = 3
    config = {
        "env": env_name,
        "num_sgd_iter": 2,
        "train_batch_size": 400,
        # "update_steps": 1000,
        # **get_marl_env_config(env_name, num_agents)
    }
    initialize_ray(test_mode=True, local_mode=True)
    train(
        EPTrainer,
        config,
        exp_name="DELETE_ME_TEST",
        stop={"timesteps_total": 10000},
        test_mode=True
    )
