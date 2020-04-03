"""
We build a DiCE trainer in this file.

In each DiCE trainer, we have multiple polices. We maintain a reference of
the whole team of polices in each policy. So that for each policy it can
query other policies' responses on a given observation.

The reference of the whole team of policies is called policy map, and it is
initialized in the setup_policies_pool function below. After each iteration,
the after_optimizer_iteration is called to update the policies map in each
policy if necessary.

We also validate the config of the DiCE trainer in this file.
"""

import copy
import logging

import ray
from ray.rllib.agents.impala.impala import validate_config as original_validate
from ray.rllib.agents.ppo.appo import APPOTrainer
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.utils import try_import_tf

from toolbox.dice.dice_model import ActorDoubleCriticNetwork
from toolbox.dies.appo_impl.constants import *
from toolbox.dies.appo_impl.dice_optimizer import AsyncSamplesOptimizer
from toolbox.dies.appo_impl.dice_policy_appo import DiCEPolicy_APPO
from toolbox.dies.appo_impl.dice_workers import SuperWorkerSet

tf = try_import_tf()

logger = logging.getLogger(__name__)
DEFAULT_POLICY_ID = "default_policy"


def validate_config(config):
    """Validate the config"""
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

    # validate other elements of IMPALA config
    original_validate(config)


def _convert_weights(weights, new_name, old_name=DEFAULT_POLICY_ID):
    assert isinstance(weights, dict)
    return {k.replace(old_name, new_name): v for k, v in weights.items()}


def _build_cloned_policy_config(config):
    policy_config = copy.deepcopy(config)
    policy_config["num_agents"] = 0
    policy_config["num_gpus"] = 0
    policy_config["num_workers"] = 0
    policy_config["_i_am_clone"] = True
    return policy_config


def setup_policies_pool(trainer):
    """Initialize the team of agents by calling the function in each policy"""
    # Target only used in vtrace
    # trainer.target_update_frequency = \
    #     trainer.config["num_sgd_iter"] * trainer.config[
    #     "minibatch_buffer_size"]

    assert len(trainer.workers.items()) == trainer.config["num_agents"]

    refer_policy = trainer.get_policy()[0]
    act_space = refer_policy.action_space
    obs_space = refer_policy.observation_space
    policy_config = _build_cloned_policy_config(trainer.config)

    # Aggregate all policy in each worker set to build central policy weight
    # storage. But we do not maintain a real polices pool in trainer.
    central_policy_weights = {}
    for ws_id, worker_set in trainer.workers.items():
        weights = worker_set.local_worker().get_weights()[DEFAULT_POLICY_ID]
        central_policy_weights[ws_id] = weights

    central_policy_weights_id = ray.put(central_policy_weights)
    trainer._central_policy_weights = central_policy_weights
    policy_class = trainer._policy

    # Maintain a copy of central policy pool in each local worker set,
    # alongside with a local policies weights storage
    for ws_id, worker_set in trainer.workers.items():
        def _setup_policy_pool(worker, worker_index):
            worker._local_policy_pool = {}
            central_weights = ray.get(central_policy_weights_id)
            for policy_id, weights in central_weights.items():
                policy_name = "workerset{}_worker{}_cloned_policy{}".format(
                    ws_id, worker_index, policy_id
                )
                logger.info("Start creating policy <{}> in worker <{}> "
                            "in workerset <{}>"
                            "".format(policy_name, worker_index, ws_id))
                with tf.variable_scope(policy_name):
                    policy = policy_class(obs_space, act_space, policy_config)
                    policy.set_weights(_convert_weights(weights, policy_name))
                    worker._local_policy_pool[policy_id] = policy

            def _init_diversity_policy(policy, my_policy_name):
                # We don't have target network at all
                # policy.update_target_network(tau=1.0)
                policy._lazy_initialize(worker._local_policy_pool)
                logger.info("Finish single task of <{}> in worker <{}> in "
                            "workerset <{}>".format(
                    my_policy_name, worker_index, ws_id))

            worker.foreach_trainable_policy(_init_diversity_policy)

        worker_set.foreach_worker_with_index(_setup_policy_pool)


def after_optimizer_iteration(trainer, fetches):
    """
    1. Receive all latest policies
    2. Update the policies in local trainer
    3. Broadcast the latest policy map to each workerset
    """
    # Receive all latest policies and update the central policy pool
    for ws_id, worker_set in trainer.workers.items():
        weights = worker_set.local_worker().get_weights()[DEFAULT_POLICY_ID]
        weights = copy.deepcopy(weights)
        if trainer.config[DELAY_UPDATE]:
            tau = trainer.config["tau"]
            trainer._central_policy_weights[ws_id] = {
                k: w * tau + (1 - tau) * old_w
                for (k, w), old_w in zip(
                    weights.items(),
                    trainer._central_policy_weights[ws_id].values())
            }
        else:
            trainer._central_policy_weights[ws_id] = weights

    central_policy_weights_id = ray.put(trainer._central_policy_weights)

    # Sync the weights in each worker
    for ws_id, worker_set in trainer.workers.items():
        def _sync_policy_pool(worker, worker_index):
            central_weights = ray.get(central_policy_weights_id)
            for policy_id, weights in central_weights.items():
                policy_name = "workerset{}_worker{}_cloned_policy{}".format(
                    ws_id, worker_index, policy_id
                )
                worker._local_policy_pool[policy_id].set_weights(
                    _convert_weights(weights, policy_name)
                )

        worker_set.foreach_worker_with_index(_sync_policy_pool)

    if trainer.config["use_kl_loss"]:
        raise NotImplementedError("KL loss not used.")


def make_aggregators_and_optimizer(workers, config):
    if config["num_aggregation_workers"] > 0:
        # Create co-located aggregator actors first for placement pref
        # aggregators = TreeAggregator.precreate_aggregators(
        #     config["num_aggregation_workers"])
        raise NotImplementedError()
    else:
        aggregators = None
    workers.add_workers(config["num_workers"])

    optimizer = AsyncSamplesOptimizer(
        workers,
        lr=config["lr"],
        num_gpus=config["num_gpus"],
        sample_batch_size=config["sample_batch_size"],
        train_batch_size=config["train_batch_size"],
        replay_buffer_num_slots=config["replay_buffer_num_slots"],
        replay_proportion=config["replay_proportion"],
        num_data_loader_buffers=config["num_data_loader_buffers"],
        max_sample_requests_in_flight_per_worker=config[
            "max_sample_requests_in_flight_per_worker"],
        broadcast_interval=config["broadcast_interval"],
        num_sgd_iter=config["num_sgd_iter"],
        minibatch_buffer_size=config["minibatch_buffer_size"],
        num_aggregation_workers=config["num_aggregation_workers"],
        learner_queue_size=config["learner_queue_size"],
        learner_queue_timeout=config["learner_queue_timeout"],
        **config["optimizer"])

    if aggregators:
        # Assign the pre-created aggregators to the optimizer
        optimizer.aggregator.init(aggregators)
    return optimizer


def make_workers(trainer, env_creator, policy, config):
    # (DICE) at the init stage, the remote workers is set to zero.
    # all workers are then setup at make_aggregators_and_optimizer
    return SuperWorkerSet(
        config["num_agents"],
        env_creator,
        policy,
        config,
        num_workers_per_set=0,
        logdir=trainer.logdir
    )


DiCETrainer_APPO = APPOTrainer.with_updates(
    name="DiCETrainer_APPO",
    default_config=dice_appo_default_config,
    default_policy=DiCEPolicy_APPO,
    get_policy_class=lambda _: DiCEPolicy_APPO,
    validate_config=validate_config,
    make_workers=make_workers,
    make_policy_optimizer=make_aggregators_and_optimizer,
    after_init=setup_policies_pool,
    after_optimizer_step=after_optimizer_iteration,
)
