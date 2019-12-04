import logging

import numpy as np
import tensorflow as tf
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG, validate_config, \
    SyncSamplesOptimizer, update_kl
from ray.rllib.agents.ppo.ppo_policy import postprocess_ppo_gae, \
    make_tf_callable, setup_mixins, kl_and_loss_stats, ppo_surrogate_loss

from toolbox.marl.adaptive_extra_loss import AdaptiveExtraLossPPOTrainer, \
    AdaptiveExtraLossPPOTFPolicy, merge_dicts, NoveltyParamMixin, mixin_list, \
    AddLossMixin, wrap_stats_fn, wrap_after_train_result
from toolbox.marl.extra_loss_ppo_trainer import NO_SPLIT_OBS, PEER_ACTION, \
    SampleBatch, extra_loss_ppo_loss, cross_policy_object_without_joint_dataset
from toolbox.marl.utils import on_train_result
from toolbox.modified_rllib.multi_gpu_optimizer import \
    LocalMultiGPUOptimizerModified

logger = logging.getLogger(__name__)

DISABLE = "disable"
DISABLE_AND_EXPAND = "disable_and_expand"
REPLAY_VALUES = "replay_values"
NO_REPLAY_VALUES = "no_replay_values"
DIVERSITY_ENCOURAGING = "diversity_encouraging"
DIVERSITY_ENCOURAGING_NO_RV = "diversity_encouraging_without_replay_values"
DIVERSITY_ENCOURAGING_DISABLE = "diversity_encouraging_disable"
DIVERSITY_ENCOURAGING_DISABLE_AND_EXPAND = \
    "diversity_encouraging_disable_and_expand"
OPTIONAL_MODES = [
    DISABLE, DISABLE_AND_EXPAND, REPLAY_VALUES, NO_REPLAY_VALUES,
    DIVERSITY_ENCOURAGING, DIVERSITY_ENCOURAGING_NO_RV,
    DIVERSITY_ENCOURAGING_DISABLE, DIVERSITY_ENCOURAGING_DISABLE_AND_EXPAND
]

ceppo_default_config = merge_dicts(
    DEFAULT_CONFIG,
    dict(
        learn_with_peers=True,
        use_joint_dataset=False,
        mode=REPLAY_VALUES,
        callbacks={"on_train_result": on_train_result}
    )
    # you should add {"on_train_result": on_train_result} to callbacks.
)


def postprocess_ceppo(policy, sample_batch, others_batches=None, episode=None):
    if not policy.loss_initialized() or policy.config[DISABLE]:
        batch = postprocess_ppo_gae(policy, sample_batch)
        if policy.config[DIVERSITY_ENCOURAGING]:
            assert not policy.config["use_joint_dataset"]
            batch[NO_SPLIT_OBS] = np.zeros_like(
                sample_batch[SampleBatch.CUR_OBS], dtype=np.float32
            )
            batch[PEER_ACTION] = np.zeros_like(
                sample_batch[SampleBatch.ACTIONS], dtype=np.float32
            )
        return batch

    batches = [postprocess_ppo_gae(policy, sample_batch)]
    for pid, (_, batch) in others_batches.items():
        if policy.config[REPLAY_VALUES]:
            # use my policy to evaluate the values of other's samples.
            batch[SampleBatch.VF_PREDS] = policy._value_batch(
                batch[SampleBatch.CUR_OBS], batch[SampleBatch.PREV_ACTIONS],
                batch[SampleBatch.PREV_REWARDS]
            )
        # use my policy to postprocess other's trajectory.
        batches.append(postprocess_ppo_gae(policy, batch))
    return SampleBatch.concat_samples(batches)


class ValueNetworkMixin2(object):
    def __init__(self, config):
        if config["use_gae"]:
            @make_tf_callable(self.get_session(), True)
            def value_batch(ob, prev_action, prev_reward):
                # We do not support recurrent network now.
                model_out, _ = self.model(
                    {
                        SampleBatch.CUR_OBS: tf.convert_to_tensor(ob),
                        SampleBatch.PREV_ACTIONS: tf.
                            convert_to_tensor(prev_action),
                        SampleBatch.PREV_REWARDS: tf.
                            convert_to_tensor(prev_reward),
                        "is_training": tf.convert_to_tensor(False),
                    }
                )
                return self.model.value_function()
        else:
            @make_tf_callable(self.get_session(), True)
            def value_batch(ob, prev_action, prev_reward):
                return tf.zeros_like(prev_reward)
        self._value_batch = value_batch


def setup_mixins_ceppo(policy, obs_space, action_space, config):
    setup_mixins(policy, obs_space, action_space, config)
    if not config['disable']:
        ValueNetworkMixin2.__init__(policy, config)
        if config[DIVERSITY_ENCOURAGING]:
            AddLossMixin.__init__(policy, config)
            NoveltyParamMixin.__init__(policy, config)


def validate_and_rewrite_config(config):
    mode = config['mode']
    assert mode in OPTIONAL_MODES

    # hyper-parameter: DIVERSITY_ENCOURAGING
    if mode in [
        DIVERSITY_ENCOURAGING,
        DIVERSITY_ENCOURAGING_NO_RV,
        DIVERSITY_ENCOURAGING_DISABLE,
        DIVERSITY_ENCOURAGING_DISABLE_AND_EXPAND
    ]:
        config[DIVERSITY_ENCOURAGING] = True
        config.update(
            novelty_loss_param_init=0.000001,
            novelty_loss_increment=10.0,
            novelty_loss_running_length=10,
            joint_dataset_sample_batch_size=200,
            novelty_mode="mean",
            use_joint_dataset=False
        )
    else:
        config[DIVERSITY_ENCOURAGING] = False

    # hyper-parameter: REPLAY_VALUES
    if mode in [
        REPLAY_VALUES,
        DIVERSITY_ENCOURAGING
    ]:
        config[REPLAY_VALUES] = True
    else:
        config[REPLAY_VALUES] = False

    # hyper-parameter: DISABLE
    if mode in [
        DISABLE,
        DISABLE_AND_EXPAND,
        DIVERSITY_ENCOURAGING_DISABLE,
        DIVERSITY_ENCOURAGING_DISABLE_AND_EXPAND
    ]:
        config[DISABLE] = True
    else:
        config[DISABLE] = False

    # DISABLE_AND_EXPAND requires to modified the config.
    if mode in [
        DISABLE_AND_EXPAND,
        DIVERSITY_ENCOURAGING_DISABLE_AND_EXPAND
    ]:
        num_agents = len(config['multiagent']['policies'])
        config['train_batch_size'] = config['train_batch_size'] * num_agents
        config['num_envs_per_worker'] = \
            config['num_envs_per_worker'] * num_agents

    # validate config
    validate_config(config)
    assert not config.get("use_joint_dataset")
    assert "callbacks" in config
    assert "on_train_result" in config['callbacks']
    assert DISABLE in config
    assert DIVERSITY_ENCOURAGING in config
    assert REPLAY_VALUES in config


def choose_policy_optimizer_modified(workers, config):
    """The original optimizer has wrong number of trained samples stats.
    So we make little modification and use the corrected optimizer."""
    if config["simple_optimizer"]:
        return SyncSamplesOptimizer(
            workers,
            num_sgd_iter=config["num_sgd_iter"],
            train_batch_size=config["train_batch_size"],
            sgd_minibatch_size=config["sgd_minibatch_size"],
            standardize_fields=["advantages"]
        )

    num_agents = len(config['multiagent']['policies'])

    if config[DISABLE]:
        compute_num_steps_sampled = None
    else:

        def compute_num_steps_sampled(batch):
            counts = np.mean([b.count for b in batch.policy_batches.values()])
            return int(counts / num_agents)

    if config[DIVERSITY_ENCOURAGING]:
        process_multiagent_batch_fn = cross_policy_object_without_joint_dataset
        no_split_list = [PEER_ACTION, NO_SPLIT_OBS]
    else:
        process_multiagent_batch_fn = None
        no_split_list = None

    return LocalMultiGPUOptimizerModified(
        workers,
        compute_num_steps_sampled=compute_num_steps_sampled,
        no_split_list=no_split_list,
        process_multiagent_batch_fn=process_multiagent_batch_fn,
        sgd_batch_size=config["sgd_minibatch_size"],
        num_sgd_iter=config["num_sgd_iter"],
        num_gpus=config["num_gpus"],
        sample_batch_size=config["sample_batch_size"],
        num_envs_per_worker=config["num_envs_per_worker"],
        train_batch_size=config["train_batch_size"],
        standardize_fields=["advantages"],
        shuffle_sequences=config["shuffle_sequences"]
    )


def wrap_stats_ceppo(policy, train_batch):
    if policy.config[DIVERSITY_ENCOURAGING]:
        return wrap_stats_fn(policy, train_batch)
    else:
        return kl_and_loss_stats(policy, train_batch)


def wrap_after_train_result_ceppo(trainer, fetches):
    if trainer.config[DIVERSITY_ENCOURAGING]:
        wrap_after_train_result(trainer, fetches)
    else:
        update_kl(trainer, fetches)


def loss_ceppo(policy, model, dist_class, train_batch):
    if policy.config[DIVERSITY_ENCOURAGING]:
        return extra_loss_ppo_loss(policy, model, dist_class, train_batch)
    else:
        return ppo_surrogate_loss(policy, model, dist_class, train_batch)


CEPPOTFPolicy = AdaptiveExtraLossPPOTFPolicy.with_updates(
    name="CEPPOTFPolicy",
    get_default_config=lambda: ceppo_default_config,
    postprocess_fn=postprocess_ceppo,
    loss_fn=loss_ceppo,
    before_loss_init=setup_mixins_ceppo,
    stats_fn=wrap_stats_ceppo,
    mixins=mixin_list + [AddLossMixin, NoveltyParamMixin, ValueNetworkMixin2]
)

CEPPOTrainer = AdaptiveExtraLossPPOTrainer.with_updates(
    name="CEPPO",
    after_optimizer_step=wrap_after_train_result_ceppo,
    default_config=ceppo_default_config,
    default_policy=CEPPOTFPolicy,
    validate_config=validate_and_rewrite_config,
    make_policy_optimizer=choose_policy_optimizer_modified
)
