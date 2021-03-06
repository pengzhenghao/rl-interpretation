from ray.rllib.agents.ppo.ppo_tf_policy import BEHAVIOUR_LOGITS, \
    PPOLoss as original_PPOLoss, ppo_surrogate_loss
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch

from toolbox.dece.dece_vtrace import build_appo_surrogate_loss
from toolbox.dece.utils import *
from ray.rllib.agents.ppo.ppo_tf_policy import BEHAVIOUR_LOGITS, \
    PPOLoss as original_PPOLoss, ppo_surrogate_loss
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch

from toolbox.dece.dece_vtrace import build_appo_surrogate_loss
from toolbox.dece.utils import *

logger = logging.getLogger(__name__)


def PPOLoss(*args, **kwargs):
    """A workaround"""
    if "is_ratio" in kwargs:
        kwargs.pop("is_ratio")
    return original_PPOLoss(*args, **kwargs)


def loss_dece(policy, model, dist_class, train_batch):
    # if policy.config[I_AM_CLONE]:  # the basic loss. useless.
    #     return tnb_loss(policy, model, dist_class, train_batch)
    if not policy.config[DIVERSITY_ENCOURAGING]:
        return ppo_surrogate_loss(policy, model, dist_class, train_batch)
    if policy.config[REPLAY_VALUES]:
        return build_appo_surrogate_loss(
            policy, model, dist_class, train_batch
        )
        # return tnb_loss(policy, model, dist_class, train_batch)
    # if policy.config[USE_BISECTOR]:
    #     return tnb_loss(policy, model, dist_class, train_batch)
    else:  # USE_BISECTOR makes difference at computing_gradient!
        # So here are same either.
        return tnb_loss(policy, model, dist_class, train_batch)


class PPOLossTwoSideNovelty(object):
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

        # surrogate_loss = tf.minimum(
        #     advantages * logp_ratio,
        #     advantages *
        #     tf.clip_by_value(logp_ratio, 1 - clip_param, 1 + clip_param)
        # )

        new_surrogate_loss = advantages * tf.minimum(
            logp_ratio, 1 + clip_param
        )

        self.mean_policy_loss = reduce_mean_valid(-new_surrogate_loss)
        self.mean_vf_loss = tf.constant(0.0)
        loss = reduce_mean_valid(
            -new_surrogate_loss + cur_kl_coeff * action_kl -
            entropy_coeff * curr_entropy
        )
        self.loss = loss


class PPOLossTwoSideClip(object):
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
            is_ratio=None
    ):
        def reduce_mean_valid(t):
            return tf.reduce_mean(tf.boolean_mask(t, valid_mask))

        prev_dist = dist_class(prev_logits, model)
        # Make loss functions.
        logp_ratio = tf.exp(curr_action_dist.logp(actions) - prev_actions_logp)
        action_kl = prev_dist.kl(curr_action_dist)
        self.mean_kl = reduce_mean_valid(action_kl)
        curr_entropy = curr_action_dist.entropy()
        self.mean_entropy = reduce_mean_valid(curr_entropy)

        new_surrogate_loss = advantages * tf.minimum(
            logp_ratio, 1 + clip_param
        )
        self.mean_policy_loss = reduce_mean_valid(-new_surrogate_loss)

        if use_gae:
            vf_loss1 = tf.square(value_fn - value_targets)
            vf_clipped = vf_preds + tf.clip_by_value(
                value_fn - vf_preds, -vf_clip_param, vf_clip_param
            )
            vf_loss2 = tf.square(vf_clipped - value_targets)
            vf_loss = tf.maximum(vf_loss1, vf_loss2)
            self.mean_vf_loss = reduce_mean_valid(vf_loss)
            loss = reduce_mean_valid(
                -new_surrogate_loss + cur_kl_coeff * action_kl +
                vf_loss_coeff * vf_loss - entropy_coeff * curr_entropy
            )
        else:
            self.mean_vf_loss = tf.constant(0.0)
            loss = reduce_mean_valid(
                -new_surrogate_loss + cur_kl_coeff * action_kl -
                entropy_coeff * curr_entropy
            )
        self.loss = loss


# def vtrace_loss(policy, model, dist_class, train_batch):


class PPOLossVtrace(object):
    """Deprecated"""

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
            is_ratio=None
    ):
        raise ValueError()
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

        prev_dist = dist_class(prev_logits, model)
        # Make loss functions.
        logp_ratio = tf.exp(curr_action_dist.logp(actions) - prev_actions_logp)
        logp_ratio = is_ratio * logp_ratio
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

        if use_gae:
            vf_loss1 = tf.square(value_fn - value_targets)
            vf_clipped = vf_preds + tf.clip_by_value(
                value_fn - vf_preds, -vf_clip_param, vf_clip_param
            )
            vf_loss2 = tf.square(vf_clipped - value_targets)
            vf_loss = tf.maximum(vf_loss1, vf_loss2)
            self.mean_vf_loss = reduce_mean_valid(vf_loss)
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
        self.loss = loss


def tnb_loss(policy, model, dist_class, train_batch):
    """Add novelty loss with original ppo loss using TNB method"""
    assert not policy.config[REPLAY_VALUES]
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
    loss_cls = PPOLossTwoSideClip \
        if policy.config[TWO_SIDE_CLIP_LOSS] else PPOLoss
    # if policy.config[REPLAY_VALUES]:
    #     loss_cls = PPOLossVtrace
    policy.loss_obj = loss_cls(
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
        model_config=policy.config["model"],
        # is_ratio=train_batch['is_ratio'] if policy.config[REPLAY_VALUES]
        # else None
    )
    # FIXME we don't prepare to use vtrace in no replay values mode.
    if policy.config[USE_DIVERSITY_VALUE_NETWORK]:
        policy.novelty_loss_obj = loss_cls(
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
            model_config=policy.config["model"],
            is_ratio=train_batch['is_ratio']
            if policy.config[REPLAY_VALUES] else None
        )
    else:
        policy.novelty_loss_obj = PPOLossTwoSideNovelty(
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
    policy.debug_ratio = train_batch["debug_ratio"]
    policy.abs_advantage = train_batch["abs_advantage"]
    return [policy.loss_obj.loss, policy.novelty_loss_obj.loss]


def _flatten(tensor):
    flat = tf.reshape(tensor, shape=[-1])
    return flat, tensor.shape, flat.shape


def tnb_gradients(policy, optimizer, loss):
    # if policy.config[I_AM_CLONE]:
    #     return optimizer.compute_gradients(loss)
    if (not policy.config[USE_BISECTOR]) or (policy.config[I_AM_CLONE]):
        with tf.control_dependencies([loss[1]]):
            policy_grad = optimizer.compute_gradients(loss[0])
        return policy_grad

    policy_grad = optimizer.compute_gradients(loss[0])
    novelty_grad = optimizer.compute_gradients(loss[1])

    # return_grads_order = []

    return_gradients = {}
    policy_grad_flatten = []
    policy_grad_info = []
    novelty_grad_flatten = []
    novelty_grad_info = []

    for (pg, var), (ng, var2) in zip(policy_grad, novelty_grad):
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

    if policy.config[CONSTRAIN_NOVELTY] == 'soft':
        tg = tf.linalg.l2_normalize(
            policy._alpha * novelty_grad_norm +
            (1 - policy._alpha) * policy_grad_norm
        )
    else:  # CONSTRAIN_NOVELTY == hard or None
        tg = tf.linalg.l2_normalize(policy_grad_norm + novelty_grad_norm)

    pg_length = tf.norm(tf.multiply(policy_grad_flatten, tg))
    ng_length = tf.norm(tf.multiply(novelty_grad_flatten, tg))

    if policy.config[CLIP_DIVERSITY_GRADIENT]:
        ng_length = tf.minimum(pg_length, ng_length)

    tg_lenth = (pg_length + ng_length) / 2
    tg = tg * tg_lenth
    if policy.config[CONSTRAIN_NOVELTY] == 'hard':
        total_grad = tf.cond(
            policy._alpha < 0.5,
            # use pure policy gradient if alpha too small
            lambda: policy_grad_flatten,
            lambda: tg
        )
    else:
        total_grad = tg

    policy.gradient_cosine_similarity = cos_similarity
    policy.policy_grad_norm = tf.norm(policy_grad_flatten)
    policy.novelty_grad_norm = tf.norm(novelty_grad_flatten)

    # reshape back the gradients
    count = 0
    for idx, (flat_shape, org_shape, var) in enumerate(policy_grad_info):
        assert flat_shape is not None
        # return_gradients.append((None, var))
        #     continue
        size = flat_shape.as_list()[0]
        grad = total_grad[count:count + size]
        return_gradients[var] = (tf.reshape(grad, org_shape), var)
        count += size

    return [return_gradients[var] for _, var in policy_grad]
