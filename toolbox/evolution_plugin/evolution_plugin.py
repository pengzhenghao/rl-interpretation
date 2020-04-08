import copy
import logging
import time

import numpy as np
import ray
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG, \
    validate_config as original_validate, PPOTFPolicy
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.memory import ray_get_and_free
from ray.rllib.utils.timer import TimerStat
from ray.tune.utils import merge_dicts

from toolbox import initialize_ray, train
from toolbox.evolution import GaussianESTrainer
from toolbox.evolution.modified_es import DEFAULT_CONFIG as es_config
from toolbox.evolution_plugin.fuse_gradient import fuse_gradient, HARD_FUSE

tf = try_import_tf()
logger = logging.getLogger(__name__)

# TODO support ARS default config
ESPlugin = GaussianESTrainer
# ARSPlugin = GaussianARSTrainer


ppo_es_default_config = merge_dicts(
    DEFAULT_CONFIG,
    dict(
        # set episodes_per_batch to 1 so that only train_batch_size control ES
        # learning steps in each epoch
        evolution=merge_dicts(es_config, dict(
            episodes_per_batch=1, train_batch_size=4000
        )),
        fuse_mode=HARD_FUSE,
        grad_clip=40
    ))


def _flatten(weights):
    flat = np.concatenate([w.flatten() for w in weights.values()])
    shape = {wid: w.shape for wid, w in weights.items()}
    return flat, shape


def _unflatten(weights, shapes):
    assert isinstance(shapes, dict)
    i = 0
    weights_dict = {}
    for wid, shape in shapes.items():
        size = np.prod(shape, dtype=np.int)
        array = weights[i:(i + size)].reshape(shape)
        weights_dict[wid] = array
        i += size
    assert len(weights) == i, "Passed weight does not have the correct shape."
    return weights_dict


def _get_diff(new_weights, old_weights):
    """Filter out value network weights, flatten the weights"""
    assert isinstance(new_weights, dict)
    assert isinstance(old_weights, dict)
    flat_new_weights, shapes_new = _flatten(_filter_weights(new_weights))
    flat_old_weights, shapes_old = _flatten(_filter_weights(old_weights))
    assert flat_new_weights.shape == flat_old_weights.shape
    _check_shapes(shapes_new, shapes_old)
    return flat_new_weights - flat_old_weights, shapes_new


def _sync_weights(trainer, plugin):
    plugin.set_weights.remote(
        _filter_weights(trainer.get_weights()))


def _filter_weights(weights):
    """Filter out the weights of value network"""
    assert isinstance(weights, dict)
    return {
        wid: w for wid, w in weights.items() if "value" not in wid
    }


def _check_shapes(shapes1, shapes2):
    # TODO remove this function
    for (n1, s1), (n2, s2) in zip(shapes1.items(), shapes2.items()):
        assert n1 == n2
        assert s1 == s2


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
        def step(self, _test_return_old_weights=False):
            if _test_return_old_weights:
                old_weights = copy.deepcopy(
                    self.get_policy().variables.get_weights())
            train_result = self.train()
            timesteps_this_iter = train_result["timesteps_this_iter"]
            train_result = train_result["info"]
            train_result["timesteps_this_iter"] = timesteps_this_iter
            weights = self.get_policy().variables.get_weights()  # dict weights
            if _test_return_old_weights:
                return train_result, weights, old_weights
            return train_result, weights

    trainer._evolution_plugin = EvolutionPluginRemote.remote(
        trainer.config["evolution"], trainer.config["env"])
    _sync_weights(trainer, trainer._evolution_plugin)

    # These three internal variables should be updated each optimization step
    trainer._previous_master_weights = trainer.get_policy().get_weights()
    trainer._evolution_start_time = time.time()
    trainer._evolution_result = trainer._evolution_plugin.step.remote()
    trainer._fuse_timer = TimerStat()


def after_optimizer_step(trainer, fetches):
    """Collect gradients, gradient fusing, update master weights, sync weights,
    launch new evolution iteration"""
    evolution_train_result, new_weights = \
        ray_get_and_free(trainer._evolution_result)
    evolution_train_result["evolution_time"] = \
        time.time() - trainer._evolution_start_time

    evolution_diff, shapes2 = _get_diff(new_weights,
                                        trainer._previous_master_weights)
    current_master_weights = trainer.get_policy().get_weights()
    master_diff, shapes = _get_diff(current_master_weights,
                                    trainer._previous_master_weights)

    _check_shapes(shapes, shapes2)

    # Compute the fuse gradient
    with trainer._fuse_timer:
        new_grad, stats = fuse_gradient(
            master_diff, evolution_diff, trainer.config["fuse_mode"],
            max_grad_norm=trainer.config["grad_clip"])
        updated_weights = _flatten(_filter_weights(
            trainer._previous_master_weights))[0] + new_grad
        updated_weights = _unflatten(updated_weights, shapes)
        assert all("value" not in k for k in updated_weights.keys())
    stats["fuse_time"] = trainer._fuse_timer.mean

    # Sync the latest weights
    trainer.get_policy().set_weights(updated_weights)
    _sync_weights(trainer, trainer._evolution_plugin)
    trainer._previous_master_weights = trainer.get_policy().get_weights()

    # Launch a new ES epoch
    trainer._evolution_result = trainer._evolution_plugin.step.remote()
    trainer._evolution_start_time = time.time()

    fetches["fuse_gradient"] = stats
    fetches["evolution"] = evolution_train_result


def sgd_optimizer(policy, config):
    print("You are using SGD optimizer!")
    return tf.train.GradientDescentOptimizer(config["lr"])


EPPolicy = PPOTFPolicy.with_updates(
    name="EvolutionPluginTFPolicy",
    get_default_config=lambda: ppo_es_default_config,
    optimizer_fn=sgd_optimizer
)

EPTrainer = PPOTrainer.with_updates(
    name="EvolutionPlugin",
    default_policy=EPPolicy,
    get_policy_class=lambda _: EPPolicy,
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
