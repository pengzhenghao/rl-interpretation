from ray.rllib.agents.ppo.ppo import PPOTrainer, LocalMultiGPUOptimizer, \
    SyncSamplesOptimizer

from toolbox.ipd.tnb_policy import TNBPolicy, NOVELTY_ADVANTAGES, \
    ipd_default_config


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


TNBTrainer = PPOTrainer.with_updates(
    name="TNBPPO",
    make_policy_optimizer=choose_policy_optimizer,
    default_config=ipd_default_config,
    default_policy=TNBPolicy
)
