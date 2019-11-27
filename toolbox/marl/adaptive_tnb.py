from toolbox.marl.adaptive_extra_loss import setup_mixins_modified, \
    DEFAULT_CONFIG, merge_dicts, LearningRateSchedule, EntropyCoeffSchedule, \
    KLCoeffMixin, ValueNetworkMixin, AddLossMixin, NoveltyParamMixin, \
    wrap_after_train_result, validate_config_basic
from toolbox.marl.extra_loss_ppo_trainer import \
    kl_and_loss_stats_without_total_loss
from toolbox.marl.task_novelty_bisector import TNBPPOTrainer, TNBPPOTFPolicy

adaptive_tnb_default_config = merge_dicts(
    DEFAULT_CONFIG,
    dict(
        novelty_loss_param_init=0.000001,
        novelty_loss_increment=10.0,
        novelty_loss_running_length=10,
        clip_novelty_gradient=True,
        use_second_component=False,
        joint_dataset_sample_batch_size=200,
        use_joint_dataset=True,
        novelty_mode="mean"
    )
)


def wrap_stats_fn(policy, train_batch):
    ret = kl_and_loss_stats_without_total_loss(policy, train_batch)
    ret.update(
        novelty_loss_param=policy.novelty_loss_param,
        novelty_target=policy.novelty_target_tensor
    )
    return ret


AdaptivePPOTFPolicy = TNBPPOTFPolicy.with_updates(
    name="AdaptivePPOTFPolicy",
    get_default_config=lambda: adaptive_tnb_default_config,
    before_loss_init=setup_mixins_modified,
    stats_fn=wrap_stats_fn,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin, AddLossMixin, NoveltyParamMixin
    ]
)

AdaptiveTNBPPOTrainer = TNBPPOTrainer.with_updates(
    name="AdaptiveTNBPPO",
    default_config=adaptive_tnb_default_config,
    default_policy=AdaptivePPOTFPolicy,
    after_optimizer_step=wrap_after_train_result,
    validate_config=validate_config_basic,
)
