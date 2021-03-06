import ray
from ray.rllib.agents.ppo.ppo import PPOTrainer, update_kl, \
    validate_config as validate_config_original
from ray.rllib.models.catalog import ModelCatalog
from ray.tune.registry import _global_registry, ENV_CREATOR

from toolbox.dece.dece_policy import DECEPolicy
from toolbox.dece.utils import *
from toolbox.task_novelty_bisector.tnb_model import ActorDoubleCriticNetwork
from toolbox.modified_rllib.multi_gpu_optimizer import \
    LocalMultiGPUOptimizerCorrectedNumberOfSampled

logger = logging.getLogger(__name__)


def validate_config(config):
    # assert REPLAY_VALUES not in config
    # config[REPLAY_VALUES] = config[REPLAY_VALUES]

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
    # PENGZH UPDATE 2019.01.12
    # We do not modify train batch size anymore, since the
    # issue in counting the SampleBatch is corrected.

    if config[REPLAY_VALUES]:
        # use vtrace, need to check sgd_minibatch_size
        assert config['sgd_minibatch_size'] % \
               (config['env_config']['num_agents'] *
                config['sample_batch_size']) == 0, \
            "sgd_minibatch_size: {}, num_agents: {}, sample_batch_size: {}" \
            "".format(config['sgd_minibatch_size'],
                      config['env_config']['num_agents'],
                      config['sample_batch_size'])
        assert config['sgd_minibatch_size'] >= (
                config['env_config']['num_agents'] * config[
            'sample_batch_size']
        )

    validate_config_original(config)

    if not config[DIVERSITY_ENCOURAGING]:
        assert not config[USE_BISECTOR]
        assert not config[USE_DIVERSITY_VALUE_NETWORK]
        # assert not config[]

    assert config[CONSTRAIN_NOVELTY] in ['soft', 'hard', None]


def make_policy_optimizer_tnbes(workers, config):
    """The original optimizer has wrong number of trained samples stats.
    So we make little modification and use the corrected optimizer.
    This function is only made for PPO.
    """
    if config["simple_optimizer"]:
        raise NotImplementedError()

    if (not config[REPLAY_VALUES]) and config[NORMALIZE_ADVANTAGE]:
        normalized_fields = ["advantages", NOVELTY_ADVANTAGES]
    else:
        normalized_fields = []

    return LocalMultiGPUOptimizerCorrectedNumberOfSampled(
        workers,
        compute_num_steps_sampled=None,
        sgd_batch_size=config["sgd_minibatch_size"],
        num_sgd_iter=config["num_sgd_iter"],
        num_gpus=config["num_gpus"],
        sample_batch_size=config["sample_batch_size"],
        num_envs_per_worker=config["num_envs_per_worker"],
        train_batch_size=config["train_batch_size"],
        standardize_fields=normalized_fields,
        shuffle_sequences=config["shuffle_sequences"]
        if not config[REPLAY_VALUES] else False
    )


def setup_policies_pool(trainer):
    """
    Three steps to initialize policies pool.
    First, sync weights of policies in local_worker.policy_map to remote
    workers.
    Second, build polices in each worker, based on the policy_map in them.
    Third, build the target model in each policy in policy_map, by syncing
    the weights of policy.model.



    """
    if (not trainer.config[DELAY_UPDATE]) or (trainer.config[I_AM_CLONE]):
        return
    assert not trainer.get_policy('agent0').initialized_policies_pool
    # first step, broadcast local weights to remote worker.
    if trainer.workers.remote_workers():
        weights = ray.put(trainer.workers.local_worker().get_weights())
        for e in trainer.workers.remote_workers():
            e.set_weights.remote(weights)
        # by doing these, we sync the worker.polices for all workers.

    def _init_pool(worker, worker_index):
        def _init_novelty_policy(policy, my_policy_name):
            policy.update_clone_network(tau=1.0)
            policy._lazy_initialize(worker.policy_map, my_policy_name)

        worker.foreach_policy(_init_novelty_policy)

    trainer.workers.foreach_worker_with_index(_init_pool)


def after_optimizer_iteration(trainer, fetches):
    """Update the policies pool in each policy."""
    update_kl(trainer, fetches)
    for pid in fetches.keys():
        fetches[pid]["novelty_target"] = \
            trainer.get_policy(pid)._novelty_target

    if trainer.config[DELAY_UPDATE] and (not trainer.config[I_AM_CLONE]):
        if trainer.workers.remote_workers():
            weights = ray.put(trainer.workers.local_worker().get_weights())
            for e in trainer.workers.remote_workers():
                e.set_weights.remote(weights)

            def _delay_update_for_worker(worker, worker_index):
                worker.foreach_policy(lambda p, _: p.update_clone_network())

            trainer.workers.foreach_worker_with_index(_delay_update_for_worker)

    if trainer.config[CONSTRAIN_NOVELTY] is not None:
        # print("***** enter update after optimizer iteration")

        def update(pi, pi_id):
            if pi_id in fetches:
                pi.update_alpha(fetches[pi_id]["novelty_reward_mean"])
            else:
                logger.debug(
                    "No data for {}, not updating alpha".format(pi_id))

        trainer.workers.foreach_worker(
            lambda w: w.foreach_trainable_policy(update))


DECETrainer = PPOTrainer.with_updates(
    name="DECETrainer",
    default_config=dece_default_config,
    default_policy=DECEPolicy,
    validate_config=validate_config,
    make_policy_optimizer=make_policy_optimizer_tnbes,
    after_init=setup_policies_pool,
    after_optimizer_step=after_optimizer_iteration,
)
