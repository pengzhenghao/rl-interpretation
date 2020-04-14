import tensorflow as tf
from ray.rllib.agents.ppo.ppo_tf_policy import SampleBatch, \
    BEHAVIOUR_LOGITS, PPOLoss
from ray.rllib.evaluation.postprocessing import Postprocessing

from toolbox.ipd.tnb_utils import *


class PPOLossNovelty(object):
    def __init__(
            self,
            dist_class,
            model,
            advantages,
            actions,
            prev_logits,
            prev_actions_logp,
            curr_action_dist,
            cur_kl_coeff,
            valid_mask,
            entropy_coeff=0,
            clip_param=0.1
    ):
        def reduce_mean_valid(t):
            return tf.reduce_mean(tf.boolean_mask(t, valid_mask))

        prev_dist = dist_class(prev_logits, model)
        logp_ratio = tf.exp(curr_action_dist.logp(actions) - prev_actions_logp)
        action_kl = prev_dist.kl(curr_action_dist)
        self.mean_kl = reduce_mean_valid(action_kl)
        curr_entropy = curr_action_dist.entropy()
        self.mean_entropy = reduce_mean_valid(curr_entropy)
        surrogate_loss = tf.minimum(
            advantages * logp_ratio,
            advantages *
            tf.clip_by_value(logp_ratio, 1 - clip_param, 1 + clip_param)
        )
        self.mean_policy_loss = reduce_mean_valid(-surrogate_loss)
        self.mean_vf_loss = tf.constant(0.0)
        loss = reduce_mean_valid(
            -surrogate_loss + cur_kl_coeff * action_kl -
            entropy_coeff * curr_entropy
        )
        self.loss = loss


def tnb_loss(policy, model, dist_class, train_batch):
    """Add novelty loss with original ppo loss using TNB method"""
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

    policy.loss_obj = PPOLoss(
        policy.action_space,
        dist_class,
        model,
        train_batch[Postprocessing.VALUE_TARGETS],
        train_batch[Postprocessing.ADVANTAGES],
        train_batch[SampleBatch.ACTIONS],
        train_batch[BEHAVIOUR_LOGITS],
        train_batch["action_logp"],
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
        model_config=policy.config["model"]
    )

    # if policy.enable_novelty:
    if policy.config['use_novelty_value_network']:
        policy.novelty_loss_obj = PPOLoss(
            policy.action_space,
            dist_class,
            model,
            train_batch[NOVELTY_VALUE_TARGETS],
            train_batch[NOVELTY_ADVANTAGES],
            train_batch[SampleBatch.ACTIONS],
            train_batch[BEHAVIOUR_LOGITS],
            train_batch["action_logp"],
            train_batch[NOVELTY_VALUES],
            action_dist,
            model.novelty_value_function(),
            policy.kl_coeff,
            mask,
            entropy_coeff=policy.entropy_coeff,
            clip_param=policy.config["clip_param"],
            vf_clip_param=policy.config["vf_clip_param"],
            vf_loss_coeff=policy.config["vf_loss_coeff"],
            use_gae=policy.config["use_gae"],
            model_config=policy.config["model"]
        )
    else:
        policy.novelty_loss_obj = PPOLossNovelty(
            dist_class,
            model,
            train_batch[NOVELTY_ADVANTAGES],
            train_batch[SampleBatch.ACTIONS],
            train_batch[BEHAVIOUR_LOGITS],
            train_batch["action_logp"],
            action_dist,
            policy.kl_coeff,
            mask,
            entropy_coeff=policy.entropy_coeff,
            clip_param=policy.config["clip_param"]
        )

    policy.novelty_reward_mean = tf.reduce_mean(train_batch[NOVELTY_REWARDS])
    policy.novelty_reward_ratio = tf.reduce_mean(
        tf.cast(
            train_batch[NOVELTY_REWARDS] > policy.config['tnb_plus_threshold'],
            'float32'
        )
    )

    return [
        policy.loss_obj.loss, policy.novelty_loss_obj.loss,
        policy.novelty_reward_mean
    ]


def _flatten(tensor):
    flat = tf.reshape(tensor, shape=[-1])
    return flat, tensor.shape, flat.shape


def tnb_gradients(policy, optimizer, loss):
    if not policy.enable_novelty:
        with tf.control_dependencies([loss[1]]):
            policy_grad = optimizer.compute_gradients(loss[0])
        return policy_grad

    policy_grad = optimizer.compute_gradients(loss[0])
    novelty_grad = optimizer.compute_gradients(loss[1])
    novelty_reward_mean = loss[2]

    return_gradients = []
    policy_grad_flatten = []
    policy_grad_info = []
    novelty_grad_flatten = []
    novelty_grad_info = []

    for (pg, var), (ng, var2) in zip(policy_grad, novelty_grad):
        assert var == var2
        if pg is None:
            return_gradients.append((ng, var))
            continue
        if ng is None:
            return_gradients.append((pg, var))
            continue

        pg_flat, pg_shape, pg_flat_shape = _flatten(pg)
        policy_grad_flatten.append(pg_flat)
        policy_grad_info.append((pg_flat_shape, pg_shape, var))

        ng_flat, ng_shape, ng_flat_shape = _flatten(ng)
        novelty_grad_flatten.append(ng_flat)
        novelty_grad_info.append((ng_flat_shape, ng_shape))

    policy_grad_flatten = tf.concat(policy_grad_flatten, 0)
    novelty_grad_flatten = tf.concat(novelty_grad_flatten, 0)

    # implement the logic of TNB
    policy_grad_norm = tf.linalg.l2_normalize(policy_grad_flatten)
    novelty_grad_norm = tf.linalg.l2_normalize(novelty_grad_flatten)
    cos_similarity = tf.reduce_sum(
        tf.multiply(policy_grad_norm, novelty_grad_norm)
    )

    def less_90_deg():
        tg = tf.linalg.l2_normalize(policy_grad_norm + novelty_grad_norm)
        pg_length = tf.norm(tf.multiply(policy_grad_flatten, tg))
        ng_length = tf.norm(tf.multiply(novelty_grad_flatten, tg))
        if hasattr(policy, "novelty_loss_param"):
            # we are not at the original TNB, at this time
            # policy.novelty_loss_param exists, we multiplied it with g_novel.
            ng_length = policy.novelty_loss_param * ng_length
        if policy.config["clip_novelty_gradient"]:
            ng_length = tf.minimum(pg_length, ng_length)
        tg_lenth = (pg_length + ng_length) / 2
        tg = tg * tg_lenth
        return tg

    def greater_90_deg():
        tg = -cos_similarity * novelty_grad_norm + policy_grad_norm
        tg = tf.linalg.l2_normalize(tg)
        tg = tg * tf.norm(tf.multiply(policy_grad_norm, tg))
        return tg

    policy.gradient_cosine_similarity = cos_similarity
    policy.policy_grad_norm = tf.norm(policy_grad_flatten)
    policy.novelty_grad_norm = tf.norm(novelty_grad_flatten)

    if policy.config["use_second_component"]:
        total_grad = tf.cond(cos_similarity > 0, less_90_deg, greater_90_deg)
    else:
        total_grad = less_90_deg()

    if policy.config['use_tnb_plus']:
        total_grad = tf.cond(
            novelty_reward_mean > policy.config['tnb_plus_threshold'],
            lambda: policy_grad_flatten,  # if true, use original policy grad.
            lambda: total_grad  # if false, use TNB
        )

    # reshape back the gradients
    count = 0
    for idx, (flat_shape, org_shape, var) in enumerate(policy_grad_info):
        if flat_shape is None:
            return_gradients.append((None, var))
            continue
        size = flat_shape.as_list()[0]
        grad = total_grad[count:count + size]
        return_gradients.append((tf.reshape(grad, org_shape), var))
        count += size

    return return_gradients
