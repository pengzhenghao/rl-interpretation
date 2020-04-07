"""
This file implement the PPO Pipeline Trainer.
"""

import logging

from ray.rllib.agents.impala.impala import validate_config as original_validate
from ray.rllib.agents.ppo.appo import DEFAULT_CONFIG as APPO_DEFAULT, \
    APPOTrainer, AsyncPPOTFPolicy
from ray.rllib.utils import try_import_tf

from toolbox.ppo_pipeline.coordinator import Coordinator
from toolbox.ppo_pipeline.pipeline import PPOPipeline
from toolbox.ppo_pipeline.ppo_loss import build_appo_surrogate_loss
from toolbox.ppo_pipeline.utils import WorkersConfig, PipelineInterface
from toolbox.utils import merge_dicts

tf = try_import_tf()

logger = logging.getLogger(__name__)

ppo_pipeline_default_config = merge_dicts(
    APPO_DEFAULT, {
        "num_agents": 1,  # Control the agent population size
        "num_sgd_iter": 10,  # In PPO this is 10
        "train_batch_size": 500,
        "sample_batch_size": 50,

        "tau": 5e-3,
        "clip_param": 0.3,

        "lr": 5e-4,
        "max_sample_requests_in_flight_per_worker": 2,  # originally 2
        "shuffle_sequences": True,
        "sgd_minibatch_size": 200,
        "sync_sampling": False,
        "vf_share_layers": False,
        "vtrace": False,

        # "replay_buffer_num_slots": 0,  # disable replay
        # "broadcast_interval": 1,
        # "num_data_loader_buffers": 1,
        # "vf_loss_coeff": 0.5,

    }
)


def make_workers(trainer, env_creator, policy, config):
    """Return a fake worker set."""
    return WorkersConfig(env_creator, policy, config, trainer.logdir)


def validate_config(config):
    """Validate the config"""
    # check the model
    config["model"]["vf_share_layers"] = config["vf_share_layers"]

    # validate other elements of IMPALA config
    original_validate(config)

    # sgd_minibatch_size should be divisible by the sample_batch_size in
    # sync mode.
    if config["sync_sampling"]:
        assert config['sgd_minibatch_size'] % (
            config['sample_batch_size']) == 0, \
            "sgd_minibatch_size: {}, num_agents: {}, sample_batch_size: {}" \
            "".format(config['sgd_minibatch_size'], config['num_agents'],
                      config['sample_batch_size'])


def make_aggregators_and_optimizer(workers_config, config):
    # if config["num_aggregation_workers"] > 0:
    #     # Create co-located aggregator actors first for placement pref
    #     # aggregators = TreeAggregator.precreate_aggregators(
    #     #     config["num_aggregation_workers"])
    #     raise NotImplementedError()
    # else:
    #     aggregators = None
    # workers.add_workers(config["num_workers"])

    PPOPipelineRemote = PPOPipeline.as_remote()

    def make_pipeline():
        # Though the input progress_callback_id is a object id,
        # in remote pipeline the progress_callback_id is automatically
        # transform to a function.
        interface = PipelineInterface.remote()
        pipeline = PPOPipelineRemote.remote(
            workers_config,
            pipeline_interface=interface,
            train_batch_size=config["train_batch_size"],
            sample_batch_size=config["sample_batch_size"],
            # lr=config["lr"],
            num_gpus=config["num_gpus"],
            replay_buffer_num_slots=config["replay_buffer_num_slots"],
            replay_proportion=config["replay_proportion"],
            num_data_loader_buffers=config["num_data_loader_buffers"],
            max_sample_requests_in_flight_per_worker=config[
                "max_sample_requests_in_flight_per_worker"],
            broadcast_interval=config["broadcast_interval"],
            num_sgd_iter=config["num_sgd_iter"],
            sgd_minibatch_size=config["sgd_minibatch_size"],
            minibatch_buffer_size=config["minibatch_buffer_size"],
            learner_queue_size=config["learner_queue_size"],
            learner_queue_timeout=config["learner_queue_timeout"],
            num_aggregation_workers=config["num_aggregation_workers"],
            shuffle_sequences=config["shuffle_sequences"],
            sync_sampling=config["sync_sampling"],
            **config["optimizer"])
        return pipeline, interface

    # TODO change num_agents to num_pipelines
    return Coordinator(workers_config, make_pipeline, config["num_agents"],
                       config["sync_sampling"])

    # if aggregators:
    #     # Assign the pre-created aggregators to the optimizer
    #     optimizer.aggregator.init(aggregators)
    # return optimizer


def initialize_target(trainer):
    if trainer.config["vtrace"]:
        trainer.workers.local_worker().foreach_trainable_policy(
            lambda p, _: p.update_target())
        trainer.target_update_frequency = \
            trainer.config["num_sgd_iter"] * trainer.config[
                "minibatch_buffer_size"]


def update_target_and_kl(trainer, fetches):
    # Update the KL coeff depending on how many steps LearnerThread has stepped
    # through
    if not trainer.config["vtrace"]:
        return
    for l_id, learner in trainer.optimizer.learner_set.items():
        learner_steps = learner.num_steps
        if learner_steps >= trainer.target_update_frequency:

            # Update Target Network
            learner.num_steps = 0
            trainer.workers.local_worker(l_id).foreach_trainable_policy(
                lambda p, _: p.update_target())

            # Also update KL Coeff
            if trainer.config["use_kl_loss"]:
                if "kl" in fetches:
                    # single-agent
                    trainer.workers.local_worker(l_id).for_policy(
                        lambda pi: pi.update_kl(fetches["kl"]))
                else:
                    def update(pi, pi_id):
                        if pi_id in fetches:
                            pi.update_kl(fetches[pi_id]["kl"])
                        else:
                            logger.debug(
                                "No data for {}, not updating kl".format(pi_id))

                    # multi-agent
                    trainer.workers.local_worker(l_id).foreach_trainable_policy(
                        update)


PPOPipelinePolicy = AsyncPPOTFPolicy.with_updates(
    name="PPOPipelineTFPolicy",
    get_default_config=lambda: ppo_pipeline_default_config,
    loss_fn=build_appo_surrogate_loss
)

PPOPipelineTrainer = APPOTrainer.with_updates(
    name="PPOPipelineTrainer",
    default_config=ppo_pipeline_default_config,
    validate_config=validate_config,
    default_policy=PPOPipelinePolicy,
    get_policy_class=lambda _: PPOPipelinePolicy,
    make_workers=make_workers,
    make_policy_optimizer=make_aggregators_and_optimizer,
    after_init=initialize_target,
    after_optimizer_step=update_target_and_kl
)
