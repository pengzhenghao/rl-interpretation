"""This file is mostly copied from RLLib"""
import tensorflow as tf
from ray.rllib.agents.ppo.ppo import PPOTFPolicy, PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import Postprocessing, SampleBatch, \
    BEHAVIOUR_LOGITS, ACTION_LOGP, explained_variance, ValueNetworkMixin, \
    LearningRateSchedule, EntropyCoeffSchedule


class PPOLossWithoutKL(object):
    def __init__(
            self,
            action_space,
            dist_class,
            model,
            value_targets,
            advantages,
            actions,
            prev_logits,
            prev_actions_logp,
            vf_preds,
            curr_action_dist,
            value_fn,
            # cur_kl_coeff,
            valid_mask,
            # entropy_coeff=0,
            clip_param=0.1,
            vf_clip_param=0.1,
            vf_loss_coeff=1.0,
            use_gae=True,
            model_config=None
    ):
        """Constructs the loss for Proximal Policy Objective.

        Arguments:
            action_space: Environment observation space specification.
            dist_class: action distribution class for logits.
            value_targets (Placeholder): Placeholder for target values; used
                for GAE.
            actions (Placeholder): Placeholder for actions taken
                from previous model evaluation.
            advantages (Placeholder): Placeholder for calculated advantages
                from previous model evaluation.
            prev_logits (Placeholder): Placeholder for logits output from
                previous model evaluation.
            prev_actions_logp (Placeholder): Placeholder for prob output from
                previous model evaluation.
            vf_preds (Placeholder): Placeholder for value function output
                from previous model evaluation.
            curr_action_dist (ActionDistribution): ActionDistribution
                of the current model.
            value_fn (Tensor): Current value function output Tensor.
            cur_kl_coeff (Variable): Variable holding the current PPO KL
                coefficient.
            valid_mask (Tensor): A bool mask of valid input elements (#2992).
            entropy_coeff (float): Coefficient of the entropy regularizer.
            clip_param (float): Clip parameter
            vf_clip_param (float): Clip parameter for the value function
            vf_loss_coeff (float): Coefficient of the value function loss
            use_gae (bool): If true, use the Generalized Advantage Estimator.
            model_config (dict): (Optional) model config for use in specifying
                action distributions.
        """

        def reduce_mean_valid(t):
            return tf.reduce_mean(tf.boolean_mask(t, valid_mask))

        # Make loss functions.
        logp_ratio = tf.exp(curr_action_dist.logp(actions) - prev_actions_logp)
        surrogate_loss = tf.minimum(
            advantages * logp_ratio,
            advantages *
            tf.clip_by_value(logp_ratio, 1 - clip_param, 1 + clip_param)
        )

        # curr_entropy = curr_action_dist.entropy()
        # self.mean_entropy = reduce_mean_valid(curr_entropy)

        self.mean_policy_loss = reduce_mean_valid(-surrogate_loss)
        if use_gae:
            vf_loss1 = tf.square(value_fn - value_targets)
            vf_clipped = vf_preds + tf.clip_by_value(
                value_fn - vf_preds, -vf_clip_param, vf_clip_param
            )
            vf_loss2 = tf.square(vf_clipped - value_targets)
            vf_loss = tf.maximum(vf_loss1, vf_loss2)
            self.mean_vf_loss = reduce_mean_valid(vf_loss)
            loss = reduce_mean_valid(-surrogate_loss + vf_loss_coeff * vf_loss)
            # - entropy_coeff * curr_entropy)
        else:
            self.mean_vf_loss = tf.constant(0.0)
            loss = reduce_mean_valid(-surrogate_loss)
            # - entropy_coeff * curr_entropy)
        self.loss = loss


def ppo_surrogate_loss_without_kl(policy, model, dist_class, train_batch):
    logits, state = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)

    if state:
        max_seq_len = tf.reduce_max(train_batch["seq_lens"])
        mask = tf.sequence_mask(train_batch["seq_lens"], max_seq_len)
        mask = tf.reshape(mask, [-1])
    else:
        mask = tf.ones_like(
            train_batch[Postprocessing.ADVANTAGES], dtype=tf.bool
        )

    policy.loss_obj = PPOLossWithoutKL(
        policy.action_space,
        dist_class,
        model,
        train_batch[Postprocessing.VALUE_TARGETS],
        train_batch[Postprocessing.ADVANTAGES],
        train_batch[SampleBatch.ACTIONS],
        train_batch[BEHAVIOUR_LOGITS],
        train_batch[ACTION_LOGP],
        train_batch[SampleBatch.VF_PREDS],
        action_dist,
        model.value_function(),
        mask,
        # entropy_coeff=policy.entropy_coeff,
        clip_param=policy.config["clip_param"],
        vf_clip_param=policy.config["vf_clip_param"],
        vf_loss_coeff=policy.config["vf_loss_coeff"],
        use_gae=policy.config["use_gae"],
        model_config=policy.config["model"]
    )

    return policy.loss_obj.loss


def loss_stats(policy, train_batch):
    return {
        "cur_lr": tf.cast(policy.cur_lr, tf.float64),
        "total_loss": policy.loss_obj.loss,
        "policy_loss": policy.loss_obj.mean_policy_loss,
        "vf_loss": policy.loss_obj.mean_vf_loss,
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS],
            policy.model.value_function()
        )
    }


def setup_mixins_without_kl(policy, obs_space, action_space, config):
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    # EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
    #                               config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])


PPOTFPolicyWithoutKL = PPOTFPolicy.with_updates(
    name="PPOTFPolicyWithoutKL",
    loss_fn=ppo_surrogate_loss_without_kl,
    stats_fn=loss_stats,
    before_loss_init=setup_mixins_without_kl,
    mixins=[
        LearningRateSchedule,
        # EntropyCoeffSchedule,
        ValueNetworkMixin
    ]
)

PPOTrainerWithoutKL = PPOTrainer.with_updates(
    name="PPOWithoutKL",
    default_policy=PPOTFPolicyWithoutKL,
    after_optimizer_step=None
)
