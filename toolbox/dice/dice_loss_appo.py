"""
The Two-side Clip Loss and the diversity-regularized learning is implemented
in this file.

First we compute the task loss and the diversity loss in dice_loss. Then we
implement the Diversity Regularization module in dice_gradient.
"""
import logging

import gym
import numpy as np
from ray.rllib.agents.ppo.appo_policy import PPOSurrogateLoss, \
    VTraceSurrogateLoss, _make_time_major, BEHAVIOUR_LOGITS
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.models.tf.tf_action_dist import Categorical
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import try_import_tf

from toolbox.dice.constants import *

tf = try_import_tf()
logger = logging.getLogger(__name__)


class PPOSurrogateDiversityLoss:
    """Diversity loss"""

    def __init__(self,
                 prev_actions_logp,
                 actions_logp,
                 action_kl,
                 actions_entropy,
                 # values,
                 valid_mask,
                 advantages,
                 # value_targets,
                 # vf_loss_coeff=0.5,
                 entropy_coeff=0.01,
                 clip_param=0.3,
                 cur_kl_coeff=None,
                 use_kl_loss=False):
        def reduce_mean_valid(t):
            return tf.reduce_mean(tf.boolean_mask(t, valid_mask))

        logp_ratio = tf.exp(actions_logp - prev_actions_logp)
        surrogate_loss = tf.minimum(
            advantages * logp_ratio,
            advantages * tf.clip_by_value(logp_ratio, 1 - clip_param,
                                          1 + clip_param))
        self.mean_kl = reduce_mean_valid(action_kl)
        self.pi_loss = -reduce_mean_valid(surrogate_loss)
        self.entropy = reduce_mean_valid(actions_entropy)
        self.total_loss = self.pi_loss - self.entropy * entropy_coeff
        if use_kl_loss:
            self.total_loss += cur_kl_coeff * self.mean_kl


def build_appo_surrogate_loss(policy, model, dist_class, train_batch):
    model_out, _ = model.from_batch(train_batch)

    # A workaround for policy that is clone
    if policy.config.get(I_AM_CLONE, False):
        return tf.reduce_sum(model_out)

    action_dist = dist_class(model_out, model)

    if isinstance(policy.action_space, gym.spaces.Discrete):
        is_multidiscrete = False
        output_hidden_shape = [policy.action_space.n]
    elif isinstance(policy.action_space,
                    gym.spaces.multi_discrete.MultiDiscrete):
        is_multidiscrete = True
        output_hidden_shape = policy.action_space.nvec.astype(np.int32)
    else:
        is_multidiscrete = False
        output_hidden_shape = 1

    def make_time_major(*args, **kw):
        return _make_time_major(policy, train_batch.get("seq_lens"), *args,
                                **kw)

    actions = train_batch[SampleBatch.ACTIONS]
    dones = train_batch[SampleBatch.DONES]
    rewards = train_batch[SampleBatch.REWARDS]
    behaviour_logits = train_batch[BEHAVIOUR_LOGITS]

    unpacked_behaviour_logits = tf.split(
        behaviour_logits, output_hidden_shape, axis=1)
    unpacked_outputs = tf.split(model_out, output_hidden_shape, axis=1)
    prev_action_dist = dist_class(behaviour_logits, policy.model)
    values = policy.model.value_function()

    policy.model_vars = policy.model.variables()

    if policy.is_recurrent():
        max_seq_len = tf.reduce_max(train_batch["seq_lens"]) - 1
        mask = tf.sequence_mask(train_batch["seq_lens"], max_seq_len)
        mask = tf.reshape(mask, [-1])
    else:
        mask = tf.ones_like(rewards)

    if policy.config["vtrace"]:
        logger.debug("Using V-Trace surrogate loss (vtrace=True)")

        policy.target_model_vars = policy.target_model.variables()

        target_model_out, _ = policy.target_model.from_batch(train_batch)
        old_policy_behaviour_logits = tf.stop_gradient(target_model_out)

        unpacked_old_policy_behaviour_logits = tf.split(
            old_policy_behaviour_logits, output_hidden_shape, axis=1)

        old_policy_action_dist = dist_class(old_policy_behaviour_logits, model)

        # Prepare actions for loss
        loss_actions = actions if is_multidiscrete else tf.expand_dims(
            actions, axis=1)

        # Prepare KL for Loss
        mean_kl = make_time_major(
            old_policy_action_dist.multi_kl(action_dist), drop_last=True)

        # If you are using vtrace, please note that normalization advantage is
        # disable.

        policy.loss = VTraceSurrogateLoss(
            actions=make_time_major(loss_actions, drop_last=True),
            prev_actions_logp=make_time_major(
                prev_action_dist.logp(actions), drop_last=True),
            actions_logp=make_time_major(
                action_dist.logp(actions), drop_last=True),
            old_policy_actions_logp=make_time_major(
                old_policy_action_dist.logp(actions), drop_last=True),
            action_kl=tf.reduce_mean(mean_kl, axis=0)
            if is_multidiscrete else mean_kl,
            actions_entropy=make_time_major(
                action_dist.multi_entropy(), drop_last=True),
            dones=make_time_major(dones, drop_last=True),
            behaviour_logits=make_time_major(
                unpacked_behaviour_logits, drop_last=True),
            old_policy_behaviour_logits=make_time_major(
                unpacked_old_policy_behaviour_logits, drop_last=True),
            target_logits=make_time_major(unpacked_outputs, drop_last=True),
            discount=policy.config["gamma"],
            rewards=make_time_major(rewards, drop_last=True),
            values=make_time_major(values, drop_last=True),
            bootstrap_value=make_time_major(values)[-1],
            dist_class=Categorical if is_multidiscrete else dist_class,
            model=policy.model,
            valid_mask=make_time_major(mask, drop_last=True),
            vf_loss_coeff=policy.config["vf_loss_coeff"],
            entropy_coeff=policy.config["entropy_coeff"],
            clip_rho_threshold=policy.config["vtrace_clip_rho_threshold"],
            clip_pg_rho_threshold=policy.config[
                "vtrace_clip_pg_rho_threshold"],
            clip_param=policy.config["clip_param"],
            cur_kl_coeff=policy.kl_coeff,
            use_kl_loss=policy.config["use_kl_loss"])
    else:
        logger.debug("Using PPO surrogate loss (vtrace=False)")

        # Prepare KL for Loss
        mean_kl = make_time_major(prev_action_dist.multi_kl(action_dist))

        advantages = train_batch[Postprocessing.ADVANTAGES]
        if policy.config[NORMALIZE_ADVANTAGE]:
            advantages = (advantages - tf.reduce_mean(advantages)) / (
                    tf.math.reduce_std(advantages) + 1e-6
            )

        policy.loss = PPOSurrogateLoss(
            prev_actions_logp=make_time_major(prev_action_dist.logp(actions)),
            actions_logp=make_time_major(action_dist.logp(actions)),
            action_kl=tf.reduce_mean(mean_kl, axis=0)
            if is_multidiscrete else mean_kl,
            actions_entropy=make_time_major(action_dist.multi_entropy()),
            values=make_time_major(values),
            valid_mask=make_time_major(mask),
            advantages=make_time_major(advantages),
            value_targets=make_time_major(
                train_batch[Postprocessing.VALUE_TARGETS]),
            vf_loss_coeff=policy.config["vf_loss_coeff"],
            entropy_coeff=policy.config["entropy_coeff"],
            clip_param=policy.config["clip_param"],
            cur_kl_coeff=policy.kl_coeff,
            use_kl_loss=policy.config["use_kl_loss"])

    # Build the loss for diversity

    # Build the loss for diversity
    if policy.config[USE_DIVERSITY_VALUE_NETWORK]:
        # We do not consider vtrace diversity loss at all
        policy.diversity_loss = PPOSurrogateLoss(
            prev_actions_logp=make_time_major(prev_action_dist.logp(actions)),
            actions_logp=make_time_major(action_dist.logp(actions)),
            action_kl=tf.reduce_mean(mean_kl, axis=0)
            if is_multidiscrete else mean_kl,
            actions_entropy=make_time_major(action_dist.multi_entropy()),
            # Diversity value
            values=make_time_major(model.diversity_value_function()),
            valid_mask=make_time_major(mask),
            # Diversity advantage
            advantages=make_time_major(train_batch[DIVERSITY_ADVANTAGES]),
            # Value target
            value_targets=make_time_major(
                train_batch[DIVERSITY_VALUE_TARGETS]),
            vf_loss_coeff=policy.config["vf_loss_coeff"],
            entropy_coeff=policy.config["entropy_coeff"],
            clip_param=policy.config["clip_param"],
            cur_kl_coeff=policy.kl_coeff,
            use_kl_loss=policy.config["use_kl_loss"])
    else:
        policy.diversity_loss = PPOSurrogateDiversityLoss(
            prev_actions_logp=make_time_major(prev_action_dist.logp(actions)),
            actions_logp=make_time_major(action_dist.logp(actions)),
            action_kl=tf.reduce_mean(mean_kl, axis=0)
            if is_multidiscrete else mean_kl,
            actions_entropy=make_time_major(action_dist.multi_entropy()),
            # Diversity value
            # values=make_time_major(model.diversity_value_function()),
            valid_mask=make_time_major(mask),
            # Diversity advantage
            advantages=make_time_major(train_batch[DIVERSITY_ADVANTAGES]),
            # Value target
            # value_targets=make_time_major(
            #     train_batch[DIVERSITY_VALUE_TARGETS]),
            # vf_loss_coeff=policy.config["vf_loss_coeff"],
            entropy_coeff=policy.config["entropy_coeff"],
            clip_param=policy.config["clip_param"],
            cur_kl_coeff=policy.kl_coeff,
            use_kl_loss=policy.config["use_kl_loss"])
    # Add the diversity reward as a stat
    policy.diversity_reward_mean = tf.reduce_mean(
        train_batch[DIVERSITY_REWARDS]
    )
    return [policy.loss.total_loss, policy.diversity_loss.total_loss]
    # return policy.loss.total_loss


def _flatten(tensor):
    flat = tf.reshape(tensor, shape=[-1])
    return flat, tensor.shape, flat.shape


def dice_gradient(policy, optimizer, loss):
    """Implement the idea of gradients bisector to fuse the task gradients
    with the diversity gradient.
    """

    # A workaround to deal with cloned policy
    if policy.config.get(I_AM_CLONE, False):
        return optimizer.compute_gradients(loss)

    if not policy.config[USE_BISECTOR]:
        # For ablation study. If don't use bisector, we simply return the
        # task gradient.
        with tf.control_dependencies([loss[1]]):
            policy_grad = optimizer.compute_gradients(loss[0])
        if policy.config["grad_clip"] is not None:
            clipped_grads, _ = tf.clip_by_global_norm([
                g for g, _ in policy_grad if g is not None
            ], policy.config["grad_clip"]
            )
            # Reorder the gradients since some are None and can't be clipped
            ret = []
            clipped_g_id = 0
            for org_g, v in policy_grad:
                if org_g is None:
                    ret.append((org_g, v))
                else:
                    if clipped_g_id >= len(clipped_grads):
                        print('ss')
                    ret.append((clipped_grads[clipped_g_id], v))
                    clipped_g_id += 1
            return ret

        else:
            return policy_grad

    policy_grad = optimizer.compute_gradients(loss[0])
    diversity_grad = optimizer.compute_gradients(loss[1])

    return_gradients = {}
    policy_grad_flatten = []
    policy_grad_info = []
    diversity_grad_flatten = []
    diversity_grad_info = []

    # First, flatten task gradient and diversity gradient into two vector.
    for (pg, var), (ng, var2) in zip(policy_grad, diversity_grad):
        assert var == var2
        if pg is None:
            return_gradients[var] = (ng, var2)
            continue
        if ng is None:
            return_gradients[var] = (pg, var)
            continue

        pg_flat, pg_shape, pg_flat_shape = _flatten(pg)
        policy_grad_flatten.append(pg_flat)
        policy_grad_info.append((pg_flat_shape, pg_shape, var))

        ng_flat, ng_shape, ng_flat_shape = _flatten(ng)
        diversity_grad_flatten.append(ng_flat)
        diversity_grad_info.append((ng_flat_shape, ng_shape))

    policy_grad_flatten = tf.concat(policy_grad_flatten, 0)
    diversity_grad_flatten = tf.concat(diversity_grad_flatten, 0)

    # Second, compute the norm of two gradient.
    policy_grad_norm = tf.linalg.l2_normalize(policy_grad_flatten)
    diversity_grad_norm = tf.linalg.l2_normalize(diversity_grad_flatten)

    # Third, compute the bisector.
    final_grad = tf.linalg.l2_normalize(policy_grad_norm + diversity_grad_norm)

    # Fourth, compute the length of the final gradient.
    pg_length = tf.norm(tf.multiply(policy_grad_flatten, final_grad))
    ng_length = tf.norm(tf.multiply(diversity_grad_flatten, final_grad))
    if policy.config[CLIP_DIVERSITY_GRADIENT]:
        ng_length = tf.minimum(pg_length, ng_length)
    tg_lenth = (pg_length + ng_length) / 2

    final_grad = final_grad * tg_lenth

    # add some stats.
    policy.gradient_cosine_similarity = tf.reduce_sum(
        tf.multiply(policy_grad_norm, diversity_grad_norm)
    )
    policy.policy_grad_norm = tf.norm(policy_grad_flatten)
    policy.diversity_grad_norm = tf.norm(diversity_grad_flatten)

    # Fifth, split the flatten vector into the original form as the final
    # gradients.
    count = 0
    for idx, (flat_shape, org_shape, var) in enumerate(policy_grad_info):
        assert flat_shape is not None
        size = flat_shape.as_list()[0]
        grad = final_grad[count:count + size]
        return_gradients[var] = (tf.reshape(grad, org_shape), var)
        count += size

    if policy.config["grad_clip"] is not None:
        ret_grads = [return_gradients[var][0] for _, var in policy_grad]
        clipped_grads, _ = tf.clip_by_global_norm(
            ret_grads, policy.config["grad_clip"])
        return [(g, return_gradients[var][1])
                for g, (_, var) in zip(clipped_grads, policy_grad)]
    else:
        return [return_gradients[var] for _, var in policy_grad]
