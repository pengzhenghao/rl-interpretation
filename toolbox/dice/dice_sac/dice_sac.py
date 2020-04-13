import ray
from ray.rllib.agents.dqn.dqn import update_target_if_needed, \
    check_config_and_setup_param_noise
from ray.rllib.agents.sac.sac import SACTrainer
from ray.rllib.models.catalog import ModelCatalog
from ray.tune.registry import _global_registry, ENV_CREATOR

import toolbox.dice.utils as constants
from toolbox.dice.dice import setup_policies_pool, make_policy_optimizer_tnbes
from toolbox.dice.dice_model import ActorDoubleCriticNetwork
from toolbox.dice.dice_sac.dice_sac_config import dice_sac_default_config
from toolbox.dice.dice_sac.dice_sac_policy import DiCESACPolicy
from toolbox.dice.utils import *


def after_optimizer_step(trainer, fetches):
    # Original SAC operation
    update_target_if_needed(trainer, fetches)

    # only update the policies pool if used DELAY_UPDATE, otherwise
    # the policies_pool in each policy is simply not used, so we don't
    # need to update it.
    if trainer.config[constants.DELAY_UPDATE]:
        if trainer.workers.remote_workers():
            weights = ray.put(trainer.workers.local_worker().get_weights())
            for e in trainer.workers.remote_workers():
                e.set_weights.remote(weights)

            def _delay_update_for_worker(worker, worker_index):
                worker.foreach_policy(lambda p, _: p.update_target_network())

            trainer.workers.foreach_worker_with_index(_delay_update_for_worker)


def validate_config(config):
    check_config_and_setup_param_noise(config)
    # create multi-agent environment
    assert _global_registry.contains(ENV_CREATOR, config["env"])
    env_creator = _global_registry.get(ENV_CREATOR, config["env"])
    tmp_env = env_creator(config["env_config"])
    config["multiagent"]["policies"] = {
        i: (None, tmp_env.observation_space, tmp_env.action_space, {})
        for i in tmp_env.agent_ids
    }
    config["multiagent"]["policy_mapping_fn"] = lambda x: x

    # check the model
    if config[USE_DIVERSITY_VALUE_NETWORK]:
        ModelCatalog.register_custom_model(
            "ActorDoubleCriticNetwork", ActorDoubleCriticNetwork
        )
        config['model']['custom_model'] = "ActorDoubleCriticNetwork"
        config['model']['custom_options'] = {
            "use_diversity_value_network": config[USE_DIVERSITY_VALUE_NETWORK]
        }
    else:
        config['model']['custom_model'] = None
        config['model']['custom_options'] = None


# TODO the policy is not finish yet.

DiCESACTrainer = SACTrainer.with_updates(
    name="DiCESACTrainer",
    default_config=dice_sac_default_config,
    default_policy=DiCESACPolicy,
    get_policy_class=lambda _: DiCESACPolicy,

    # FIXME finished but not tested
    after_init=setup_policies_pool,
    after_optimizer_step=after_optimizer_step,

    # FIXME not finish
    validate_config=validate_config,
    make_policy_optimizer=make_policy_optimizer_tnbes,
)
