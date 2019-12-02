import logging

import tensorflow as tf
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.rllib.agents.ppo.ppo_policy import Postprocessing, SampleBatch, \
    BEHAVIOUR_LOGITS, ACTION_LOGP, postprocess_ppo_gae, PPOTFPolicy, \
    setup_mixins, make_tf_callable, EntropyCoeffSchedule, ValueNetworkMixin, \
    LearningRateSchedule, KLCoeffMixin
from ray.tune.util import merge_dicts

logger = logging.getLogger(__name__)

ceppo_default_config = merge_dicts(
    DEFAULT_CONFIG, dict(
        learn_with_peers=True,
        use_myself_vf_preds=True,
        use_joint_dataset=False,
        disable=False
    )
)

required_peer_data_keys = {
    Postprocessing.VALUE_TARGETS, Postprocessing.ADVANTAGES,
    SampleBatch.ACTIONS, BEHAVIOUR_LOGITS, ACTION_LOGP, SampleBatch.VF_PREDS
}


def postprocess_ceppo(policy, sample_batch, others_batches=None, epidose=None):
    if not policy.loss_initialized() or policy.config['disable']:
        # Only for initialization. GAE postprocess done in the following func.
        return postprocess_ppo_gae(policy, sample_batch)

    if not policy.config["use_myself_vf_preds"]:
        batch = SampleBatch.concat_samples(
            [sample_batch] + [b for (_, b) in others_batches.values()])
        return postprocess_ppo_gae(policy, batch)

    # use_myself_vf_preds
    assert policy.config["use_myself_vf_preds"]
    batches = [postprocess_ppo_gae(policy, sample_batch)]
    for pid, (_, batch) in others_batches.items():
        batch[SampleBatch.VF_PREDS] = policy._value_batch(
            batch[SampleBatch.CUR_OBS], batch[SampleBatch.PREV_ACTIONS],
            batch[SampleBatch.PREV_REWARDS])
        # use my policy to postprocess other's trajectory.
        batches.append(postprocess_ppo_gae(policy, batch))
    return SampleBatch.concat_samples(batches)


class ValueNetworkMixin2(object):
    def __init__(self, config):
        self._update_value_batch_function(config['use_gae'])

    def _update_value_batch_function(self, use_gae=None):
        if use_gae is None:
            use_gae = self.config['use_gae']
        if use_gae:
            @make_tf_callable(self.get_session(), True)
            def value_batch(ob, prev_action, prev_reward):
                # We do not support recurrent network now.
                model_out, _ = self.model({
                    SampleBatch.CUR_OBS: tf.convert_to_tensor(ob),
                    SampleBatch.PREV_ACTIONS: tf.convert_to_tensor(
                        prev_action),
                    SampleBatch.PREV_REWARDS: tf.convert_to_tensor(
                        prev_reward),
                    "is_training": tf.convert_to_tensor(False),
                })
                return self.model.value_function()
        else:
            @make_tf_callable(self.get_session(), True)
            def value_batch(ob, prev_action, prev_reward):
                return tf.zeros_like(prev_reward)

        self._value_batch = value_batch


def setup_mixins_modified(policy, obs_space, action_space, config):
    ValueNetworkMixin2.__init__(policy, config)
    setup_mixins(policy, obs_space, action_space, config)


CEPPOTFPolicy = PPOTFPolicy.with_updates(
    name="CEPPOTFPolicy",
    get_default_config=lambda: ceppo_default_config,
    postprocess_fn=postprocess_ceppo,
    before_loss_init=setup_mixins_modified,
    mixins=[LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
            ValueNetworkMixin, ValueNetworkMixin2]
)

CEPPOTrainer = PPOTrainer.with_updates(
    name="CEPPO",
    default_config=ceppo_default_config,
    default_policy=CEPPOTFPolicy
)
