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

from ray.rllib.agents.impala.impala import validate_config as original_validate
from ray.rllib.agents.ppo.appo import APPOTrainer, update_kl
# update_target_and_kl as original_after_optimizer_step
from ray.rllib.models.catalog import ModelCatalog

from toolbox.dice.appo_impl.dice_optimizer import AsyncSamplesOptimizer
from toolbox.dice.appo_impl.dice_policy_appo import DiCEPolicy_APPO
from toolbox.dice.appo_impl.utils import dice_appo_default_config
from toolbox.dice.dice_model import ActorDoubleCriticNetwork
# from toolbox.dice.dice_policy import DiCEPolicy
from toolbox.dice.utils import *

# validate_config as validate_config_original
logger = logging.getLogger(__name__)


def validate_config(config):
    """Validate the config"""

    # create multi-agent environment

    # Do not using multiple policies anymore.

    # assert _global_registry.contains(ENV_CREATOR, config["env"])
    # env_creator = _global_registry.get(ENV_CREATOR, config["env"])
    # tmp_env = env_creator(config["env_config"])
    # config["multiagent"]["policies"] = {
    #     i: (None, tmp_env.observation_space, tmp_env.action_space, {})
    #     for i in tmp_env.agent_ids
    # }
    # config["multiagent"]["policy_mapping_fn"] = lambda x: x

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


# TODO You need to make clear whether advantage normalization is used or not

# def make_policy_optimizer_tnbes(workers, config):
#     """We implement the knob of NORMALIZE_ADVANTAGE here."""
#     if config["simple_optimizer"]:
#         raise NotImplementedError()
#
#     if config[NORMALIZE_ADVANTAGE]:
#         normalized_fields = ["advantages", DIVERSITY_ADVANTAGES]
#     else:
#         normalized_fields = []
#
#     return LocalMultiGPUOptimizer(
#         workers,
#         sgd_batch_size=config["sgd_minibatch_size"],
#         num_sgd_iter=config["num_sgd_iter"],
#         num_gpus=config["num_gpus"],
#         sample_batch_size=config["sample_batch_size"],
#         num_envs_per_worker=config["num_envs_per_worker"],
#         train_batch_size=config["train_batch_size"],
#         standardize_fields=normalized_fields,
#         shuffle_sequences=config["shuffle_sequences"]
#     )


def setup_policies_pool(trainer):
    """Initialize the team of agents by calling the function in each policy"""

    # FIXME temporarily disable this function
    # return

    # Original initialization
    # TODO should we maintain the concept of target network?? How we deal
    #  with the policy map?
    # The next line is not commented in original code
    # trainer.workers.local_worker().foreach_trainable_policy(
    #     lambda p, _: p.update_target())
    trainer.target_update_frequency = trainer.config["num_sgd_iter"] \
                                      * trainer.config["minibatch_buffer_size"]

    # FIXME
    return



    if not trainer.config[DELAY_UPDATE]:
        return
    assert not trainer.get_policy().initialized_policies_pool
    # First step, broadcast local weights to remote worker.
    # assert trainer.workers.remote_workers()

    # TODO this checking is necessary, but not now. please uncomment next line
    # assert len(trainer.workers.remote_workers()) == trainer.config[
    # "num_agents"]

    trainer._policy_worker_mapping = {}

    def _get_weight(worker, worker_index):
        return worker.get_weights(), worker_index

    result = trainer.workers.foreach_worker_with_index(_get_weight)
    for weights, worker_id in result:
        trainer._policy_worker_mapping[worker_id] = weights

    # print('skdjflkadsjf')
    # weights = ray.put(trainer.workers.local_worker().get_weights())
    # for e in trainer.workers.remote_workers():
    #     e.set_weights.remote(weights)

    # Second step, call the _lazy_initialize function of each policy, feeding
    # with the policies map in the trainer.
    # def _init_pool(worker, worker_index):
    # def _init_diversity_policy(policy, my_policy_name):
    #     policy.update_target_network(tau=1.0)
    #     policy._lazy_initialize(worker.policy_map, my_policy_name)

    # worker.foreach_policy(_init_diversity_policy)
    # trainer.workers.foreach_worker_with_index(_init_pool)


def after_optimizer_iteration(trainer, fetches):
    """Update the policies pool in each policy."""
    # original_after_optimizer_step(trainer, fetches)  # original APPO procedure

    # Original APPO
    # Update the KL coeff depending on how many steps LearnerThread has stepped
    # through
    # TODO Wrong!
    learner_steps = trainer.optimizer.learner_set[0].num_steps
    # learner_steps = trainer.optimizer.learner.num_steps
    if learner_steps >= trainer.target_update_frequency:

        # Update Target Network
        for learner in trainer.optimizer.learner_set.values():
            learner.num_steps = 0
        trainer.workers.local_worker().foreach_trainable_policy(
            lambda p, _: p.update_target())

        # Also update KL Coeff
        if trainer.config["use_kl_loss"]:
            # TODO wrong!
            # update_kl(trainer, trainer.optimizer.learner.stats)
            update_kl(trainer, trainer.optimizer.learner_set[0].stats)

    # only update the policies pool if used DELAY_UPDATE, otherwise
    # the policies_pool in each policy is simply not used, so we don't
    # need to update it.
    # if trainer.config[DELAY_UPDATE]:
    #     if trainer.workers.remote_workers():
    #         weights = ray.put(trainer.workers.local_worker().get_weights())
    #         for e in trainer.workers.remote_workers():
    #             e.set_weights.remote(weights)
    #
    #         def _delay_update_for_worker(worker, worker_index):
    #             worker.foreach_policy(lambda p, _: p.update_target_network())
    #
    #         trainer.workers.foreach_worker_with_index(
    #         _delay_update_for_worker)


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


from toolbox.dice.appo_impl.dice_workers import SuperWorkerSet


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
