import logging

import gym
from ray.rllib.agents.impala import vtrace
from ray.rllib.agents.impala.vtrace_policy import BEHAVIOUR_LOGITS
from ray.rllib.models.tf.tf_action_dist import Categorical
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import try_import_tf

from toolbox.dece.utils import *

tf = try_import_tf()

POLICY_SCOPE = "func"
TARGET_POLICY_SCOPE = "target_func"

logger = logging.getLogger(__name__)


class VTraceSurrogateLoss(object):
    def __init__(self,
                 actions,
                 prev_actions_logp,
                 actions_logp,
                 # old_policy_actions_logp,
                 action_kl,
                 actions_entropy,
                 dones,
                 behaviour_logits,
                 # old_policy_behaviour_logits,
                 target_logits,
                 discount,
                 rewards,
                 values,
                 bootstrap_value,
                 dist_class,
                 model,
                 valid_mask,
                 vf_loss_coeff=0.5,
                 entropy_coeff=0.01,
                 clip_rho_threshold=1.0,
                 clip_pg_rho_threshold=1.0,
                 clip_param=0.3,
                 cur_kl_coeff=None,
                 use_kl_loss=False):
        """APPO Loss, with IS modifications and V-trace for Advantage
        Estimation

        VTraceLoss takes tensors of shape [T, B, ...], where `B` is the
        batch_size. The reason we need to know `B` is for V-trace to properly
        handle episode cut boundaries.

        Arguments:
            actions: An int|float32 tensor of shape [T, B, logit_dim].
            prev_actions_logp: A float32 tensor of shape [T, B].
            actions_logp: A float32 tensor of shape [T, B].
            old_policy_actions_logp: A float32 tensor of shape [T, B].
            action_kl: A float32 tensor of shape [T, B].
            actions_entropy: A float32 tensor of shape [T, B].
            dones: A bool tensor of shape [T, B].
            behaviour_logits: A float32 tensor of shape [T, B, logit_dim].
            old_policy_behaviour_logits: A float32 tensor of shape
            [T, B, logit_dim].
            target_logits: A float32 tensor of shape [T, B, logit_dim].
            discount: A float32 scalar.
            rewards: A float32 tensor of shape [T, B].
            values: A float32 tensor of shape [T, B].
            bootstrap_value: A float32 tensor of shape [B].
            dist_class: action distribution class for logits.
            model: backing ModelV2 instance
            valid_mask: A bool tensor of valid RNN input elements (#2992).
            vf_loss_coeff (float): Coefficient of the value function loss.
            entropy_coeff (float): Coefficient of the entropy regularizer.
            clip_param (float): Clip parameter.
            cur_kl_coeff (float): Coefficient for KL loss.
            use_kl_loss (bool): If true, use KL loss.
        """

        def reduce_mean_valid(t):
            return tf.reduce_mean(tf.boolean_mask(t, valid_mask))

        # Compute vtrace on the CPU for better perf.
        with tf.device("/cpu:0"):
            self.vtrace_returns = vtrace.multi_from_logits(
                behaviour_policy_logits=behaviour_logits,
                target_policy_logits=target_logits,
                actions=tf.unstack(actions, axis=2),
                discounts=tf.to_float(~dones) * discount,
                rewards=rewards,
                values=values,
                bootstrap_value=bootstrap_value,
                dist_class=dist_class,
                model=model,
                clip_rho_threshold=tf.cast(clip_rho_threshold, tf.float32),
                clip_pg_rho_threshold=tf.cast(clip_pg_rho_threshold,
                                              tf.float32))

        self.is_ratio = tf.clip_by_value(
            tf.exp(prev_actions_logp - actions_logp), 0.0, 2.0)
        logp_ratio = self.is_ratio * tf.exp(actions_logp - prev_actions_logp)

        advantages = self.vtrace_returns.pg_advantages

        advantages = (advantages - tf.reduce_mean(advantages)
                        ) / max(1e-4, tf.math.reduce_std(advantages))

        self.advantage = advantages
        self.debug_ratio = logp_ratio
        surrogate_loss = tf.minimum(
            advantages * logp_ratio,
            advantages * tf.clip_by_value(logp_ratio, 1 - clip_param,
                                          1 + clip_param))

        self.mean_kl = reduce_mean_valid(action_kl)
        self.mean_policy_loss = -reduce_mean_valid(surrogate_loss)

        # The baseline loss
        delta = values - self.vtrace_returns.vs
        self.value_targets = self.vtrace_returns.vs
        self.mean_vf_loss = 0.5 * reduce_mean_valid(tf.square(delta))

        # The entropy loss
        self.mean_entropy = reduce_mean_valid(actions_entropy)

        # The summed weighted loss
        self.loss = (
                self.mean_policy_loss + self.mean_vf_loss * vf_loss_coeff -
                self.mean_entropy * entropy_coeff)

        # Optional additional KL Loss
        if use_kl_loss:
            self.loss += cur_kl_coeff * self.mean_kl


def _make_time_major(policy, seq_lens, tensor, drop_last=False):
    """Swaps batch and trajectory axis.

    Arguments:
        policy: Policy reference
        seq_lens: Sequence lengths if recurrent or None
        tensor: A tensor or list of tensors to reshape.
        drop_last: A bool indicating whether to drop the last
        trajectory item.

    Returns:
        res: A tensor with swapped axes or a list of tensors with
        swapped axes.
    """
    if isinstance(tensor, list):
        return [
            _make_time_major(policy, seq_lens, t, drop_last) for t in tensor
        ]

    if policy.is_recurrent():
        B = tf.shape(seq_lens)[0]
        T = tf.shape(tensor)[0] // B
    else:
        # Important: chop the tensor into batches at known episode cut
        # boundaries. TODO(ekl) this is kind of a hack
        T = policy.config["sample_batch_size"]
        B = tf.shape(tensor)[0] // T
        # B = 1
        # T = tf.shape(tensor)[0]

    rs = tf.reshape(tensor, tf.concat([[B, T], tf.shape(tensor)[1:]], axis=0))

    # swap B and T axes
    res = tf.transpose(
        rs, [1, 0] + list(range(2, 1 + int(tf.shape(tensor).shape[0]))))

    if drop_last:
        return res[:-1]
    return res


def build_appo_surrogate_loss(policy, model, dist_class, train_batch):
    model_out, _ = model.from_batch(train_batch)
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

    # PENGZH: Compared to APPO, we also have three concepts need to be
    # clarified here:
    #   1. old_policy, which is a target_network at APPO, is 'my policy' in
    #       our setting. Namely, the network that hasn't updated yet.
    #   2. action_dist, the latest network that is updating, so it's a few
    #       steps ahead of old_policy.
    #   3. prev_action_dist, the action distribution that 'other policy'
    #       generated, it belongs to 'other policy'.

    # PENGZH: 20200104 13:33:28
    # We argue that the above statements are not correct.
    # 1. old_policy: deprecated, we treat it = target_policy = curr_policy
    # 2. prev_dist = train_batch[logit] = other policy

    # target_model_out, _ = policy.target_model.from_batch(train_batch)
    # old_policy_behaviour_logits = tf.stop_gradient(target_model_out)
    # old_policy_behaviour_logits = train_batch["my_policy_logits"]
    actions = train_batch[SampleBatch.ACTIONS]
    rewards = train_batch[SampleBatch.REWARDS]
    behaviour_logits = train_batch[BEHAVIOUR_LOGITS]

    unpacked_behaviour_logits = tf.split(
        behaviour_logits, output_hidden_shape, axis=1)
    # unpacked_old_policy_behaviour_logits = tf.split(
    #     old_policy_behaviour_logits, output_hidden_shape, axis=1)
    unpacked_outputs = tf.split(model_out, output_hidden_shape, axis=1)
    # old_policy_action_dist = dist_class(old_policy_behaviour_logits, model)
    prev_action_dist = dist_class(behaviour_logits, policy.model)
    # values = policy.model.value_function()

    # policy.model_vars = policy.model.variables()
    # policy.target_model_vars = policy.target_model.variables()

    if policy.is_recurrent():
        max_seq_len = tf.reduce_max(train_batch["seq_lens"]) - 1
        mask = tf.sequence_mask(train_batch["seq_lens"], max_seq_len)
        mask = tf.reshape(mask, [-1])
    else:
        mask = tf.ones_like(rewards)

    assert policy.config["use_vtrace"]
    logger.debug("Using V-Trace surrogate loss (vtrace=True)")

    # Prepare KL for Loss
    # compared the updated network and not-updated network's KL.
    mean_kl = make_time_major(
        action_dist.multi_kl(action_dist), drop_last=True)

    # Prepare actions for loss
    loss_actions = actions if is_multidiscrete else tf.expand_dims(
        actions, axis=1)
    loss_actions = make_time_major(loss_actions, drop_last=True)
    prev_actions_logp = make_time_major(
            prev_action_dist.logp(actions), drop_last=True)
    actions_logp = make_time_major(action_dist.logp(actions), drop_last=True)
    action_kl = tf.reduce_mean(mean_kl, axis=0) \
        if is_multidiscrete else mean_kl
    actions_entropy = make_time_major(
            action_dist.multi_entropy(), drop_last=True)
    dones = make_time_major(train_batch[SampleBatch.DONES], drop_last=True)
    loss_behaviour_logits = make_time_major(
        unpacked_behaviour_logits, drop_last=True)
    loss_target_logits = make_time_major(unpacked_outputs, drop_last=True)
    loss_mask = make_time_major(mask, drop_last=True)

    values = model.value_function()
    # with tf.control_dependencies(assert_list):
    policy.loss_obj = VTraceSurrogateLoss(
        actions=loss_actions,
        prev_actions_logp=prev_actions_logp,
        actions_logp=actions_logp,
        # old_policy_actions_logp=make_time_major(
        #     old_policy_action_dist.logp(actions), drop_last=True),
        # TODO can be simplify
        action_kl=action_kl,
        actions_entropy=actions_entropy,
        dones=dones,
        behaviour_logits=loss_behaviour_logits,
        # old_policy_behaviour_logits=make_time_major(
        #     unpacked_old_policy_behaviour_logits, drop_last=True),
        target_logits=loss_target_logits,
        discount=policy.config["gamma"],

        rewards=make_time_major(rewards, drop_last=True),
        values=make_time_major(values, drop_last=True),
        bootstrap_value=make_time_major(values)[-1],

        dist_class=Categorical if is_multidiscrete else dist_class,
        model=policy.model,
        valid_mask=loss_mask,
        vf_loss_coeff=policy.config["vf_loss_coeff"],
        entropy_coeff=policy.config["entropy_coeff"],
        clip_rho_threshold=policy.config["clip_rho_threshold"],
        clip_pg_rho_threshold=policy.config[ "clip_pg_rho_threshold"],
        clip_param=policy.config["clip_param"],
        cur_kl_coeff=policy.kl_coeff,
        # use_kl_loss=policy.config["use_kl_loss"]
        use_kl_loss=False
    )

    novelty_values = model.novelty_value_function()
    novelty_reward = train_batch[NOVELTY_REWARDS]
    policy.novelty_loss_obj = VTraceSurrogateLoss(
        actions=loss_actions,
        prev_actions_logp=prev_actions_logp,
        actions_logp=actions_logp,
        # old_policy_actions_logp=make_time_major(
        #     old_policy_action_dist.logp(actions), drop_last=True),
        action_kl=action_kl,
        actions_entropy=actions_entropy,
        dones=dones,
        behaviour_logits=loss_behaviour_logits,
        # old_policy_behaviour_logits=make_time_major(
        #     unpacked_old_policy_behaviour_logits, drop_last=True),
        target_logits=loss_target_logits,
        discount=policy.config["gamma"],

        rewards=make_time_major(novelty_reward, drop_last=True),
        values=make_time_major(novelty_values, drop_last=True),
        bootstrap_value=make_time_major(novelty_values)[-1],

        dist_class=Categorical if is_multidiscrete else dist_class,
        model=policy.model,
        valid_mask=loss_mask,
        vf_loss_coeff=policy.config["vf_loss_coeff"],
        entropy_coeff=policy.config["entropy_coeff"],
        clip_rho_threshold=policy.config["clip_rho_threshold"],
        clip_pg_rho_threshold=policy.config["clip_pg_rho_threshold"],
        clip_param=policy.config["clip_param"],
        cur_kl_coeff=policy.kl_coeff,
        # use_kl_loss=policy.config["use_kl_loss"])
        use_kl_loss=False
    )

    policy.novelty_reward_mean = tf.reduce_mean(train_batch[NOVELTY_REWARDS])
    policy.debug_ratio = policy.loss_obj.debug_ratio
    policy.abs_advantage = tf.abs(policy.loss_obj.advantage)
    return [policy.loss_obj.loss, policy.novelty_loss_obj.loss]
