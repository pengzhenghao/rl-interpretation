import logging

import tensorflow as tf
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG, \
    validate_config
from ray.rllib.agents.ppo.ppo_policy import SampleBatch, \
    postprocess_ppo_gae, PPOTFPolicy, \
    setup_mixins, make_tf_callable, EntropyCoeffSchedule, ValueNetworkMixin, \
    LearningRateSchedule, KLCoeffMixin
from ray.tune.util import merge_dicts

logger = logging.getLogger(__name__)

DISABLE = "disable"
DISABLE_AND_EXPAND = "disable_and_expand"
REPLAY_VALUES = "replay_values"
NO_REPLAY_VALUES = "no_replay_values"
OPTIONAL_MODES = [DISABLE, DISABLE_AND_EXPAND, REPLAY_VALUES, NO_REPLAY_VALUES]

ceppo_default_config = merge_dicts(
    DEFAULT_CONFIG,
    dict(
        learn_with_peers=True,
        use_joint_dataset=False,
        mode=REPLAY_VALUES
    )
)


def postprocess_ceppo(policy, sample_batch, others_batches=None, epidose=None):
    if not policy.loss_initialized() or policy.config['disable']:
        return postprocess_ppo_gae(policy, sample_batch)

    batches = [postprocess_ppo_gae(policy, sample_batch)]
    for pid, (_, batch) in others_batches.items():
        if policy.config["use_myself_vf_preds"]:
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


def setup_mixins_modified(policy, obs_space, action_space, config):
    ValueNetworkMixin2.__init__(policy, config)
    setup_mixins(policy, obs_space, action_space, config)


def validate_and_rewrite_config(config):
    validate_config(config)

    mode = config['mode']
    assert mode in OPTIONAL_MODES
    if mode == REPLAY_VALUES:
        config['use_myself_vf_preds'] = True
    else:
        config['use_myself_vf_preds'] = False

    if mode in [DISABLE, DISABLE_AND_EXPAND]:
        config['disable'] = True
    else:
        config['disable'] = False

    if mode == DISABLE_AND_EXPAND:
        num_agents = len(config['multiagent']['policies'])
        config['train_batch_size'] = \
            ceppo_default_config['train_batch_size'] * num_agents
        config['num_workers'] = \
            ceppo_default_config['num_workers'] * num_agents


CEPPOTFPolicy = PPOTFPolicy.with_updates(
    name="CEPPOTFPolicy",
    get_default_config=lambda: ceppo_default_config,
    postprocess_fn=postprocess_ceppo,
    before_loss_init=setup_mixins_modified,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin, ValueNetworkMixin2
    ]
)

CEPPOTrainer = PPOTrainer.with_updates(
    name="CEPPO",
    default_config=ceppo_default_config,
    default_policy=CEPPOTFPolicy,
    validate_config=validate_and_rewrite_config
)
