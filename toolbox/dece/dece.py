import logging

from ray.rllib.agents.ppo.ppo import PPOTrainer, update_kl, \
    validate_config as validate_config_original
from ray.tune.registry import _global_registry, ENV_CREATOR

from toolbox.dece.dece_policy import DECEPolicy
from toolbox.dece.utils import *
from toolbox.dece.utils import _I_AM_CLONE
from toolbox.modified_rllib.multi_gpu_optimizer import \
    LocalMultiGPUOptimizerCorrectedNumberOfSampled

logger = logging.getLogger(__name__)


def validate_config(config):
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
    if config[DIVERSITY_ENCOURAGING] and config[USE_DIVERSITY_VALUE_NETWORK]:
        ModelCatalog.register_custom_model(
            "ActorDoubleCriticNetwork", ActorDoubleCriticNetwork
        )

        config['model']['custom_model'] = "ActorDoubleCriticNetwork"
        config['model']['custom_options'] = {
            "use_novelty_value_network": config[USE_DIVERSITY_VALUE_NETWORK]
            # the name 'novelty' is deprecated
        }
    else:
        config['model']['custom_model'] = None
        config['model']['custom_options'] = None

    # Reduce the train batch size for each agent
    num_agents = len(config['multiagent']['policies'])
    config['train_batch_size'] = int(config['train_batch_size'] // num_agents)
    assert config['train_batch_size'] >= config["sgd_minibatch_size"]

    validate_config_original(config)

    if not config[DIVERSITY_ENCOURAGING]:
        assert not config[USE_BISECTOR]
        assert not config[USE_DIVERSITY_VALUE_NETWORK]
        # assert not config[]


def make_policy_optimizer_tnbes(workers, config):
    """The original optimizer has wrong number of trained samples stats.
    So we make little modification and use the corrected optimizer.
    This function is only made for PPO.
    """
    if config["simple_optimizer"]:
        raise NotImplementedError()

    return LocalMultiGPUOptimizerCorrectedNumberOfSampled(
        workers,
        compute_num_steps_sampled=None,
        sgd_batch_size=config["sgd_minibatch_size"],
        num_sgd_iter=config["num_sgd_iter"],
        num_gpus=config["num_gpus"],
        sample_batch_size=config["sample_batch_size"],
        num_envs_per_worker=config["num_envs_per_worker"],
        train_batch_size=config["train_batch_size"],
        standardize_fields=["advantages", NOVELTY_ADVANTAGES],  # HERE!
        shuffle_sequences=config["shuffle_sequences"]
    )


def _delay_update(trainer, tau=None):
    weights = {k: p.get_weights() for k, p in
               trainer.workers.local_worker().policy_map.items()}

    def _func(worker):
        def _func_policy(policy, my_policy_name):
            policy._delay_update(weights, my_policy_name, tau=tau)

        worker.foreach_policy(_func_policy)

    trainer.workers.foreach_worker(_func)


def setup_policies_pool(trainer):
    if (not trainer.config[DELAY_UPDATE]) or trainer.config[_I_AM_CLONE]:
        return
    assert not trainer.get_policy('agent0').initialized_policies_pool
    weights = {k: p.get_weights() for k, p in
               trainer.workers.local_worker().policy_map.items()}

    def _init_pool(worker):
        def _init_novelty_policy(policy, my_policy_name):
            policy._lazy_initialize(weights, my_policy_name)

        worker.foreach_policy(_init_novelty_policy)

    trainer.workers.foreach_worker(_init_pool)
    # _delay_update(trainer, 1.0)


def after_optimizer_iteration(trainer, fetches):
    """Update the policies pool in each policy."""
    update_kl(trainer, fetches)
    if trainer.config[DELAY_UPDATE] and (not trainer.config[_I_AM_CLONE]):
        _delay_update(trainer)


DECETrainer = PPOTrainer.with_updates(
    name="DECETrainer",
    default_config=dece_default_config,
    default_policy=DECEPolicy,
    validate_config=validate_config,
    make_policy_optimizer=make_policy_optimizer_tnbes,
    after_init=setup_policies_pool,
    after_optimizer_step=after_optimizer_iteration
)
