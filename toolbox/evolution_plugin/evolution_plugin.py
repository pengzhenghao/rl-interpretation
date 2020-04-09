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
from ray.tune.resources import Resources
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
            episodes_per_batch=1,
            train_batch_size=4000,
            num_cpus_per_worker=0.5,
            # faster if more workers are used
            num_workers=10,  # 6 CPU for evolution plugin
            optimizer_type="sgd",  # must in [adam, sgd]
        )),
        grad_clip=40,

        # must in [HARD_FUSE, SOFT_FUSE]
        fuse_mode=HARD_FUSE,

        # must in [adam, sgd]
        master_optimizer_type="sgd",

        # if True, then force evolution diff and master diff to have equal norm
        equal_norm=False
    )
)


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
            ret_train_result = train_result["info"]
            ret_train_result["timesteps_this_iter"] = train_result[
                "timesteps_this_iter"]
            ret_train_result["timesteps_total"] = train_result[
                "timesteps_total"]
            ret_train_result["episode_reward_mean"] = train_result[
                "episode_reward_mean"]
            weights = self.get_policy().variables.get_weights()  # dict weights
            if _test_return_old_weights:
                return ret_train_result, weights, old_weights
            return ret_train_result, weights

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
    # Receive the evolution result and store the latest plugin weights
    evolution_train_result, new_weights = \
        ray_get_and_free(trainer._evolution_result)
    trainer.state["plugin"] = ray_get_and_free(
        trainer._evolution_plugin.__getstate__.remote())
    evolution_train_result["evolution_time"] = \
        time.time() - trainer._evolution_start_time

    # Compute the difference compared to master weights
    evolution_diff, shapes2 = _get_diff(
        new_weights, trainer._previous_master_weights)
    current_master_weights = trainer.get_policy().get_weights()
    master_diff, shapes = _get_diff(
        current_master_weights, trainer._previous_master_weights)
    _check_shapes(shapes, shapes2)

    # Compute the fuse gradient
    with trainer._fuse_timer:
        new_grad, stats = fuse_gradient(
            master_diff, evolution_diff, trainer.config["fuse_mode"],
            max_grad_norm=trainer.config["grad_clip"],
            equal_norm=trainer.config["equal_norm"]
        )
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
    fetches["evolution"] = evolution_train_result
    fetches["fuse_gradient"] = stats


def sgd_optimizer(policy, config):
    if config["master_optimizer_type"] == "adam":
        logger.info("You are using Adam optimizer!")
        return tf.train.AdamOptimizer(config["lr"])
    elif config["master_optimizer_type"] == "sgd":
        logger.info("You are using SGD optimizer!")
        return tf.train.GradientDescentOptimizer(config["lr"])
    else:
        raise ValueError("master_optimizer_type must in [adam, sgd].")


class OverrideDefaultResourceRequest:
    """Copied from IMPALA trainer. Add evolution plugin CPUs to original."""

    @classmethod
    def default_resource_request(cls, config):
        cf = merge_dicts(cls._default_config, config)
        return Resources(
            cpu=cf["num_cpus_for_driver"] + cf["evolution"][
                "num_cpus_for_driver"],
            gpu=cf["num_gpus"],
            memory=cf["memory"],
            object_store_memory=cf["object_store_memory"],
            extra_cpu=cf["num_cpus_per_worker"] * cf["num_workers"] +
                      cf["evolution"]["num_cpus_per_worker"] * cf["evolution"][
                          "num_workers"],  # <<== Add plugin CPUs
            extra_gpu=cf["num_gpus_per_worker"] * cf["num_workers"],
            extra_memory=cf["memory_per_worker"] * cf["num_workers"],
            extra_object_store_memory=cf["object_store_memory_per_worker"] *
                                      cf["num_workers"])


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
    validate_config=validate_config,
    mixins=[OverrideDefaultResourceRequest]
)

if __name__ == '__main__':
    config = {"env": "CartPole-v0", "num_sgd_iter": 2, "train_batch_size": 400,
              "evolution": {"num_workers": 3}}
    initialize_ray(test_mode=True, local_mode=True)
    train(EPTrainer, config, exp_name="DELETE_ME_TEST", test_mode=True,
          stop={"timesteps_total": 10000})
