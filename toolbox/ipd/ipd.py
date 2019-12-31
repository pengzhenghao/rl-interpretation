import logging

from ray.rllib.agents.ppo.ppo import PPOTrainer, LocalMultiGPUOptimizer, \
    SyncSamplesOptimizer, PPOTFPolicy, DEFAULT_CONFIG

from toolbox.ipd.tnb_policy import setup_mixins_tnb, AgentPoolMixin, \
    KLCoeffMixin, EntropyCoeffSchedule, LearningRateSchedule, \
    ValueNetworkMixin, NOVELTY_ADVANTAGES, merge_dicts

logger = logging.getLogger(__name__)


def on_episode_step(info):
    episode = info['episode']
    print('pass')


ipd_default_config = merge_dicts(
    DEFAULT_CONFIG,
    {
        "checkpoint_dict": "{}",
        "novelty_threshold": 0.5,

        # don't touch
        "use_novelty_value_network": False,
        "callbacks": {
            "on_episode_step": on_episode_step
        },
        "distance_mode": "min"
    }
)


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


IPDPolicy = PPOTFPolicy.with_updates(
    name="IPDPolicy",
    get_default_config=lambda: ipd_default_config,
    before_loss_init=setup_mixins_tnb,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin, AgentPoolMixin
    ]
)

IPDTrainer = PPOTrainer.with_updates(
    name="IPD",
    make_policy_optimizer=choose_policy_optimizer,
    default_config=ipd_default_config,
    before_train_step=before_train_step,
    default_policy=IPDPolicy
)

if __name__ == '__main__':
    from ray import tune

    from toolbox import initialize_ray

    initialize_ray(test_mode=True, local_mode=True)
    env_name = "CartPole-v0"
    config = {"num_sgd_iter": 2, "env": env_name}
    tune.run(
        IPDTrainer,
        name="DELETEME_TEST",
        verbose=2,
        stop={"timesteps_total": 50000},
        config=config
    )
