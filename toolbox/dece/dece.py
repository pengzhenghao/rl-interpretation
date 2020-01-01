import copy
import logging

import ray
from ray.rllib.agents.ppo.ppo import PPOTrainer, update_kl, \
    validate_config as validate_config_original
from ray.tune.registry import _global_registry, ENV_CREATOR

from toolbox.dece.dece_policy import DECEPolicy
from toolbox.dece.utils import *
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
    local_worker = trainer.workers.local_worker()
    if tau is None:
        tau = trainer.config['tau']

    weights = {}
    for policy_name, policy in local_worker.policy_map.items():
        weight = policy.get_weights()
        new_weight = (weight * tau + local_worker.policies_pool[
            policy_name].get_weights() * (1 - tau))
        weights[policy_name] = new_weight
        print("Successfully update the <{}> policy in local worker policies "
              "pool. Current tau: {}".format(policy_name, tau))

    weights = ray.put(weights)

    def _delay_update_for_worker(worker, worker_index):
        local_weights = ray.get(weights)
        for policy_name, tmp_weight in local_weights.items():
            worker.policies_pool[policy_name].set_weights(tmp_weight)
            print("Successfully update the <{}> policy in worker <{}> "
                  "policies pool. Current tau: {}".format(
                policy_name, "local0" if not worker_index
                else "remote{}".format(worker_index), tau))

    trainer.workers.foreach_worker_with_index(_delay_update_for_worker)
    ray.internal.free([weights])


def setup_policies_pool(trainer):
    if not trainer.config[DELAY_UPDATE]:
        return
    assert not trainer.get_policy('agent0').initialized_policies_pool
    weights = {k: p.get_weights() for k, p in
               trainer.workers.local_worker().policy_map.items()}
    weights = ray.put(weights)

    def _init_pool(worker):
        """We load the policies pool at each worker, instead of each policy,
        to save memory."""
        local_weights = ray.get(weights)
        tmp_policy = next(iter(worker.policy_map.values()))
        policies_pool = {}
        for agent_name, agent_weight in local_weights.items():
            tmp_config = copy.deepcopy(tmp_policy.config)
            # disable the private worker of each policy, to save resource.
            tmp_config.update({
                "num_workers": 0,
                "num_cpus_per_worker": 0,
                "num_cpus_for_driver": 0.2,
                "num_gpus": 0.1,
                DELAY_UPDATE: False
            })
            # build the policy and restore the weights.
            with tf.variable_scope("polices_pool/" + agent_name,
                                   reuse=tf.AUTO_REUSE):
                policy = DECEPolicy(
                    tmp_policy.observation_space, tmp_policy.action_space,
                    tmp_config
                )
                policy.set_weights(agent_weight)
            policies_pool[agent_name] = policy
        worker.policies_pool = policies_pool  # add new attribute to worker

        def _init_novelty_policy(policy, my_policy_name):
            policy._lazy_initialize(worker.policies_pool, my_policy_name)

        worker.foreach_policy(_init_novelty_policy)

    trainer.workers.foreach_worker(_init_pool)
    ray.internal.free([weights])


def after_optimizer_iteration(trainer, fetches):
    """Update the policies pool in each policy."""
    update_kl(trainer, fetches)
    if trainer.config[DELAY_UPDATE]:
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
