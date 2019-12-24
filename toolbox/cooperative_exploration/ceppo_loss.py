import tensorflow as tf
from ray.rllib.agents.ppo.ppo_policy import BEHAVIOUR_LOGITS, Postprocessing, \
    ACTION_LOGP, SampleBatch

from toolbox.cooperative_exploration.ceppo_debug import validate_tensor
from toolbox.cooperative_exploration.utils import DIVERSITY_ENCOURAGING, \
    CURIOSITY
from toolbox.marl.extra_loss_ppo_trainer import extra_loss_ppo_loss, \
    novelty_loss_mse


def loss_ceppo(policy, model, dist_class, train_batch):
    if policy.config[DIVERSITY_ENCOURAGING]:
        return extra_loss_ppo_loss(policy, model, dist_class, train_batch)
    if policy.config[CURIOSITY]:
        # create policy.novelty_loss for adaptively adjust curiosity.
        novelty_loss_mse(policy, model, dist_class, train_batch)
    return ppo_surrogate_loss(policy, model, dist_class, train_batch)


class PPOLoss(object):
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
            cur_kl_coeff,
            valid_mask,
            entropy_coeff=0,
            clip_param=0.1,
            vf_clip_param=0.1,
            vf_loss_coeff=1.0,
            use_gae=True,
            model_config=None,
            validate_nan=False
    ):
        print("Enter PPOLoss Class")
        self.stats = {}

        def validate_and_stat(tensor, name):
            tensor = validate_tensor(tensor, name, validate_nan)
            self.stats[name] = tensor
            return tensor

        value_targets = validate_and_stat(value_targets, "value_targets")
        _ = validate_and_stat(tf.square(value_targets), "value_targets_square")
        prev_logits = validate_tensor(prev_logits, "prev_logits")
        prev_actions_logp = validate_and_stat(
            prev_actions_logp, "prev_actions_logp"
        )
        vf_preds = validate_and_stat(vf_preds, "vf_preds")
        _ = validate_and_stat(tf.square(vf_preds), "vf_preds_square")
        value_fn = validate_tensor(value_fn, "value_fn")
        _ = validate_and_stat(tf.square(value_fn), "value_fn_square")

        def reduce_mean_valid(t):
            return tf.reduce_mean(tf.boolean_mask(t, valid_mask))

        prev_dist = dist_class(prev_logits, model)
        curr_action_logp = curr_action_dist.logp(actions)
        prev_actions_logp = validate_and_stat(
            prev_actions_logp, "prev_actions_logp"
        )
        curr_action_logp = validate_and_stat(
            curr_action_logp, "curr_actions_logp"
        )
        logp_diff = validate_and_stat(
            curr_action_logp - prev_actions_logp, "logp_diff"
        )
        logp_ratio = tf.exp(logp_diff)
        logp_ratio = validate_and_stat(logp_ratio, "logp_ratio")

        action_kl = prev_dist.kl(curr_action_dist)
        curr_entropy = curr_action_dist.entropy()
        self.mean_kl = validate_and_stat(reduce_mean_valid(action_kl), "kl")
        self.mean_entropy = validate_and_stat(
            reduce_mean_valid(curr_entropy), "entropy"
        )

        surrogate_loss = tf.minimum(
            advantages * logp_ratio,
            advantages *
            tf.clip_by_value(logp_ratio, 1 - clip_param, 1 + clip_param)
        )

        self.mean_policy_loss = validate_and_stat(
            reduce_mean_valid(-surrogate_loss), "policy_loss"
        )

        if use_gae:
            vf_loss1 = tf.square(value_fn - value_targets)
            vf_clipped = vf_preds + tf.clip_by_value(
                value_fn - vf_preds, -vf_clip_param, vf_clip_param
            )
            vf_loss2 = tf.square(vf_clipped - value_targets)
            vf_loss = tf.maximum(vf_loss1, vf_loss2)

            self.mean_vf_loss = validate_and_stat(
                reduce_mean_valid(vf_loss), "mean_vf_loss"
            )
            _ = validate_and_stat(vf_loss1, "vf_loss1")
            _ = validate_and_stat(vf_loss2, "vf_loss2")
            _ = validate_and_stat(vf_clipped, "vf_loss2_clipped")

            loss = reduce_mean_valid(
                -surrogate_loss + cur_kl_coeff * action_kl +
                vf_loss_coeff * vf_loss - entropy_coeff * curr_entropy
            )
        else:
            self.mean_vf_loss = tf.constant(0.0)
            loss = reduce_mean_valid(
                -surrogate_loss + cur_kl_coeff * action_kl -
                entropy_coeff * curr_entropy
            )

        self.loss = validate_and_stat(loss, "loss")


def ppo_surrogate_loss(policy, model, dist_class, train_batch):
    train_batch[
        SampleBatch.CUR_OBS
    ] = validate_tensor(train_batch[SampleBatch.CUR_OBS], "CUR_OBS")

    logits, state = model.from_batch(train_batch)
    logits = validate_tensor(logits, "action_dist logits")
    action_dist = dist_class(logits, model)
    action_dist.std = validate_tensor(action_dist.std, "action_dist.std")

    if state:
        max_seq_len = tf.reduce_max(train_batch["seq_lens"])
        mask = tf.sequence_mask(train_batch["seq_lens"], max_seq_len)
        mask = tf.reshape(mask, [-1])
    else:
        mask = tf.ones_like(
            train_batch[Postprocessing.ADVANTAGES], dtype=tf.bool
        )

    policy.loss_obj = PPOLoss(
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
        policy.kl_coeff,
        mask,
        entropy_coeff=policy.entropy_coeff,
        clip_param=policy.config["clip_param"],
        vf_clip_param=policy.config["vf_clip_param"],
        vf_loss_coeff=policy.config["vf_loss_coeff"],
        use_gae=policy.config["use_gae"],
        model_config=policy.config["model"],
        validate_nan=policy.config["check_nan"]
    )

    # add some tensors into stat
    policy.loss_obj.stats["adv_unnorm"] = \
        train_batch[Postprocessing.ADVANTAGES + "_unnormalized"]
    policy.loss_obj.stats["adv_unnorm_square"] = tf.square(
        train_batch[Postprocessing.ADVANTAGES + "_unnormalized"]
    )
    policy.loss_obj.stats["debug_fake_adv"] = train_batch['debug_fake_adv']
    policy.loss_obj.stats["debug_ratio"] = train_batch['debug_ratio']

    return policy.loss_obj.loss
