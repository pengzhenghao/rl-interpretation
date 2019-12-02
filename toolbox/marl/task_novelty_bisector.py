"""This file implement the modified version of TNB."""
import tensorflow as tf

from toolbox.marl.extra_loss_ppo_trainer import novelty_loss, \
    ppo_surrogate_loss, DEFAULT_CONFIG, merge_dicts, ExtraLossPPOTrainer, \
    ExtraLossPPOTFPolicy, kl_and_loss_stats_without_total_loss, \
    validate_config_basic

tnb_ppo_default_config = merge_dicts(
    DEFAULT_CONFIG,
    dict(
        joint_dataset_sample_batch_size=200,
        use_joint_dataset=True,
        novelty_mode="mean",
        use_second_component=True,  # whether to apply the >90deg operation
        clip_novelty_gradient=False  # whether to constraint length of g_novel
    )
)


def tnb_loss(policy, model, dist_class, train_batch):
    """Add novelty loss with original ppo loss using TNB method"""
    original_loss = ppo_surrogate_loss(policy, model, dist_class, train_batch)
    nov_loss = novelty_loss(policy, model, dist_class, train_batch)
    # In rllib convention, loss_fn should return one single tensor
    # however, there is no explicit bugs happen returning a list.
    return [original_loss, nov_loss]


def _flatten(tensor):
    flat = tf.reshape(tensor, shape=[-1])
    return flat, tensor.shape, flat.shape


def tnb_gradients(policy, optimizer, loss):
    err_msg = "We detect the {} contains less than 2 elements. " \
              "It contain only {} elements, which is not possible." \
              " Please check the codes."

    policy_grad = optimizer.compute_gradients(loss[0])
    novelty_grad = optimizer.compute_gradients(loss[1])

    # flatten the policy_grad
    policy_grad_flatten = []
    policy_grad_info = []
    for idx, (pg, var) in enumerate(policy_grad):
        if novelty_grad[idx][0] is None:
            # Some variables do not related to novelty, so the grad is None
            policy_grad_info.append((None, None, var))
            continue
        pg_flat, pg_shape, pg_flat_shape = _flatten(pg)
        policy_grad_flatten.append(pg_flat)
        policy_grad_info.append((pg_flat_shape, pg_shape, var))
    if len(policy_grad_flatten) < 2:
        raise ValueError(
            err_msg.format("policy_grad_flatten", len(policy_grad_flatten))
        )
    policy_grad_flatten = tf.concat(policy_grad_flatten, 0)

    # flatten the novelty grad
    novelty_grad_flatten = []
    novelty_grad_info = []
    for ng, _ in novelty_grad:
        if ng is None:
            novelty_grad_info.append((None, None))
            continue
        pg_flat, pg_shape, pg_flat_shape = _flatten(ng)
        novelty_grad_flatten.append(pg_flat)
        novelty_grad_info.append((pg_flat_shape, pg_shape))
    if len(novelty_grad_flatten) < 2:
        raise ValueError(
            err_msg.format("novelty_grad_flatten", len(novelty_grad_flatten))
        )
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
    policy.novelty_grad_norm = tf.norm(novelty_grad_norm)

    if policy.config["use_second_component"]:
        total_grad = tf.cond(cos_similarity > 0, less_90_deg, greater_90_deg)
    else:
        total_grad = less_90_deg()

    # reshape back the gradients
    return_gradients = []
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


def grad_stats_fn(policy, batch, grads):
    ret = {
        "cos_similarity": policy.gradient_cosine_similarity,
        "policy_grad_norm": policy.policy_grad_norm,
        "novelty_grad_norm": policy.novelty_grad_norm
    }
    return ret


TNBPPOTFPolicy = ExtraLossPPOTFPolicy.with_updates(
    name="TNBPPOTFPolicy",
    get_default_config=lambda: tnb_ppo_default_config,
    loss_fn=tnb_loss,
    gradients_fn=tnb_gradients,
    stats_fn=kl_and_loss_stats_without_total_loss,
    grad_stats_fn=grad_stats_fn
)

TNBPPOTrainer = ExtraLossPPOTrainer.with_updates(
    name="TNBPPO",
    default_config=tnb_ppo_default_config,
    validate_config=validate_config_basic,
    default_policy=TNBPPOTFPolicy
)
