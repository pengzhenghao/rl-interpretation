import numpy as np

from toolbox.cooperative_exploration.ceppo import ValueNetworkMixin2, \
    ceppo_default_config, postprocess_ceppo, _rewrite_config
from toolbox.marl.adaptive_extra_loss import AdaptiveExtraLossPPOTrainer, \
    AdaptiveExtraLossPPOTFPolicy, adaptive_extra_loss_ppo_default_config, \
    merge_dicts, setup_mixins_modified, NoveltyParamMixin, \
    validate_config_basic
from toolbox.marl.extra_loss_ppo_trainer import NO_SPLIT_OBS, \
    PEER_ACTION, SampleBatch, mixin_list, AddLossMixin

deceppo_default_config = merge_dicts(
    adaptive_extra_loss_ppo_default_config, ceppo_default_config
)


def postprocess_deceppo(
        policy, sample_batch, others_batches=None, episode=None
):
    # Replay to collect values, if applicable (mode!=disable)
    batch = postprocess_ceppo(policy, sample_batch, others_batches, episode)

    if (not policy.loss_initialized()) and (not policy.config.get('disable')):
        # To create placeholders, we create some fake data when initializing.
        if policy.config["use_joint_dataset"]:
            raise NotImplementedError()
            # batch[JOINT_OBS] = np.zeros_like(
            #     sample_batch[SampleBatch.CUR_OBS], dtype=np.float32
            # )
        else:
            batch[NO_SPLIT_OBS] = np.zeros_like(
                sample_batch[SampleBatch.CUR_OBS], dtype=np.float32
            )
        batch[PEER_ACTION] = np.zeros_like(
            sample_batch[SampleBatch.ACTIONS], dtype=np.float32
        )  # peer_action is needed no matter use joint_dataset or not.
    return batch


def setup_mixins_deceppo(policy, obs_space, action_space, config):
    ValueNetworkMixin2.__init__(policy, config)
    setup_mixins_modified(policy, obs_space, action_space, config)


def validate_and_rewrite_config(config):
    validate_config_basic(config)
    _rewrite_config(config)
    assert not config["use_joint_dataset"]  # disable this function in DECEPPO
    assert 'callbacks' in config
    assert 'on_train_result' in config['callbacks']


DECEPPOTFPolicy = AdaptiveExtraLossPPOTFPolicy.with_updates(
    name="DECEPPOTFPolicy",
    get_default_config=lambda: deceppo_default_config,
    before_loss_init=setup_mixins_deceppo,
    postprocess_fn=postprocess_deceppo,
    mixins=mixin_list + [AddLossMixin, NoveltyParamMixin, ValueNetworkMixin2]
)

# Diversity-encouraging Cooperative Exploration
DECEPPOTrainer = AdaptiveExtraLossPPOTrainer.with_updates(
    name="DECEPPO",
    default_config=deceppo_default_config,
    default_policy=DECEPPOTFPolicy,
    validate_config=validate_and_rewrite_config
)
