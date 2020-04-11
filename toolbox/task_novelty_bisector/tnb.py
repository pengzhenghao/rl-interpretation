import logging

from ray.rllib.agents.ppo.ppo import PPOTrainer, LocalMultiGPUOptimizer, \
    SyncSamplesOptimizer, validate_config as validate_config_original

from toolbox.task_novelty_bisector.tnb_policy import NOVELTY_ADVANTAGES, \
    tnb_default_config, TNBPolicy

logger = logging.getLogger(__name__)


def choose_policy_optimizer(workers, config):
    if config["simple_optimizer"]:
        return SyncSamplesOptimizer(
            workers,
            num_sgd_iter=config["num_sgd_iter"],
            train_batch_size=config["train_batch_size"],
            sgd_minibatch_size=config["sgd_minibatch_size"],
            standardize_fields=["advantages", NOVELTY_ADVANTAGES]
        )  # Here!

    return LocalMultiGPUOptimizer(
        workers,
        sgd_batch_size=config["sgd_minibatch_size"],
        num_sgd_iter=config["num_sgd_iter"],
        num_gpus=config["num_gpus"],
        sample_batch_size=config["sample_batch_size"],
        num_envs_per_worker=config["num_envs_per_worker"],
        train_batch_size=config["train_batch_size"],
        standardize_fields=["advantages", NOVELTY_ADVANTAGES],  # Here!
        shuffle_sequences=config["shuffle_sequences"]
    )


def before_train_step(trainer):
    policy = trainer.get_policy()
    if not policy.initialized_policies_pool:
        # function to call for each worker (including remote and local workers)
        def init_novelty(worker):
            # function for each policy within one worker.
            def _init_novelty_policy(policy, _):
                policy._lazy_initialize()

            worker.foreach_policy(_init_novelty_policy)

        trainer.workers.foreach_worker(init_novelty)


def validate_config(config):
    validate_config_original(config)
    assert config['model']['custom_model'] == "ActorDoubleCriticNetwork"
    config['model']['custom_options'] = {
        "use_novelty_value_network": config['use_novelty_value_network']
    }


TNBTrainer = PPOTrainer.with_updates(
    name="TNBPPO",
    validate_config=validate_config,
    make_policy_optimizer=choose_policy_optimizer,
    default_config=tnb_default_config,
    before_train_step=before_train_step,
    default_policy=TNBPolicy
)
