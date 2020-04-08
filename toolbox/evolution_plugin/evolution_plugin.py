import copy
import time

import numpy as np
import ray
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG, \
    validate_config as original_validate
from ray.rllib.utils.memory import ray_get_and_free
from ray.tune.utils import merge_dicts

from toolbox import initialize_ray, train
from toolbox.evolution import GaussianESTrainer
from toolbox.evolution.modified_es import DEFAULT_CONFIG as es_config
from toolbox.evolution_plugin.fuse_gradient import fuse_gradient, HARD_FUSE

# TODO support ARS default config
ESPlugin = GaussianESTrainer
# ARSPlugin = GaussianARSTrainer


ppo_es_default_config = merge_dicts(
    DEFAULT_CONFIG,
    dict(evolution=es_config, fuse_mode=HARD_FUSE, grad_clip=40)
)


def validate_config(config):
    original_validate(config)
    config["model"]["vf_share_layers"] = config["vf_share_layers"]
    config["evolution"]["model"] = copy.deepcopy(config["model"])
    assert not config["evolution"]["model"]["vf_share_layers"]


def after_init(trainer):
    """Setup evolution plugin, sync weight and start first training."""
    base = ESPlugin

    @ray.remote(num_gpus=0)
    class EvolutionPluginRemote(base):
        def sync_weights(self, weights):
            self.get_policy().variables.set_weights(weights)

        def retrieve_weights(self):
            return self.get_policy().variables.get_weights()

        def step(self, _test_return_old_weights=False):
            if _test_return_old_weights:
                old_weights = copy.deepcopy(self.get_weights())
            train_result = self.train()
            weights = self.get_policy().get_weights()  # flatten weights
            if _test_return_old_weights:
                return train_result, weights, old_weights
            return train_result, weights

    trainer._evolution_plugin = EvolutionPluginRemote.remote(
        trainer.config["evolution"], trainer.config["env"])
    _sync_weights(trainer, trainer._evolution_plugin)

    # These three internal variables should be updated each optimization step
    trainer._previous_master_weights = _get_flat(trainer)
    trainer._evolution_start_time = time.time()
    trainer._evolution_result = trainer._evolution_plugin.step.remote()


def after_optimizer_step(trainer, fetches):
    """Collect gradients, gradient fusing, update master weights, sync weights,
    launch new evolution iteration"""
    evolution_train_result, new_weights = \
        ray_get_and_free(trainer._evolution_result)

    evolution_diff = trainer._previous_master_weights - new_weights

    current_master_weights = _get_flat(trainer)
    master_diff = trainer._previous_master_weights - current_master_weights

    # Compute the fuse gradient
    new_grad, stats = fuse_gradient(
        master_diff, evolution_diff, trainer.config["fuse_mode"],
        max_grad_norm=trainer.config["grad_clip"])
    updated_weights = trainer._previous_master_weights + new_grad

    # Sync the latest weights
    _set_flat(trainer, updated_weights)
    _sync_weights(trainer, trainer._evolution_plugin)
    trainer._previous_master_weights = _get_flat(trainer)

    # Launch a new ES epoch
    trainer._evolution_result = trainer._evolution_plugin.step.remote()

    stats["evolution_time"] = time.time() - trainer._evolution_start_time
    fetches["fuse_gradient"] = stats
    trainer._evolution_start_time = time.time()


def _sync_weights(trainer, plugin):
    plugin.sync_weights.remote(
        filter_weights(trainer.get_policy().get_weights())
    )


def _get_flat(trainer):
    """Return the flatten weights"""
    return trainer.get_policy()._variables.get_flat()


def _set_flat(trainer, weights):
    assert isinstance(weights, np.ndarray)
    assert weights.ndim == 1
    trainer.get_policy()._variables.set_flat(weights)


def filter_weights(weights):
    """Filter out the weights of value network"""
    assert isinstance(weights, dict)
    return {
        wid: w for wid, w in weights.items() if "value" not in wid
    }


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
