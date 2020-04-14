from ray.rllib.agents.sac.sac_policy import SACTFPolicy, TargetNetworkMixin, \
    ActorCriticOptimizerMixin, ComputeTDErrorMixin, SampleBatch, \
    actor_critic_loss as sac_loss, get_dist_class
from ray.rllib.utils.tf_ops import make_tf_callable

from toolbox.dice.dice_policy import grad_stats_fn, \
    DiversityValueNetworkMixin, \
    ComputeDiversityMixin
from toolbox.dice.dice_postprocess import BEHAVIOUR_LOGITS, \
    MY_LOGIT
from toolbox.dice.dice_sac.dice_sac_config import dice_sac_default_config
from toolbox.dice.dice_sac.dice_sac_gradient import dice_sac_gradient
from toolbox.dice.utils import *


class PPOLossTwoSideDiversity:
    """Compute the PPO loss for diversity without diversity value network"""

    def __init__(
            self,
            advantages,
            policy,
            actions,
            obs,
            # current_actions_logp,
            prev_actions_logp,
            valid_mask,
            clip_param,
            policies_pool
            # entropy_coeff=0.01
    ):
        # cur_kl_coeff = 0.0

        def reduce_mean_valid(t):
            return tf.reduce_mean(tf.boolean_mask(t, valid_mask))

        # prev_dist = dist_class(prev_logits, model)

        # This is important
        # logp_ratio = tf.exp(curr_action_dist.logp(actions) -
        # prev_actions_logp)
        # model_out = policy.model.output

        # A workaround, since the model out is the flatten observation

        # ==========
        # In this part, I want to compute the diversity by myself

        my_actions = tf.split(
            policy.model.get_policy_output(obs),
            2, axis=1)[0]

        losses = []
        for p in policies_pool.values():
            logit = p.target_model.get_policy_output(obs)
            other_act = tf.split(logit, 2, axis=1)[0]
            losses.append(
                tf.keras.losses.mean_squared_error(other_act, my_actions))
        mse_loss = tf.reduce_mean(losses)
        self.loss = mse_loss
        return

        # ==========

        # This is the PPO style loss
        model_out = obs
        distribution_inputs = policy.model.get_policy_output(model_out)
        action_dist_class = get_dist_class(policy.config, policy.action_space)
        curr_action_dist = action_dist_class(distribution_inputs,
                                             policy.model)
        current_actions_logp = curr_action_dist.logp(actions)
        logp_ratio = tf.exp(current_actions_logp - prev_actions_logp)
        with tf.control_dependencies(
                [tf.check_numerics(logp_ratio, "logp_ratio")]):
            new_surrogate_loss = advantages * tf.minimum(
                logp_ratio, 1 + clip_param
            )

        # action_kl = prev_dist.kl(curr_action_dist)
        # self.mean_kl = reduce_mean_valid(action_kl)
        # curr_entropy = curr_action_dist.entropy()
        # self.mean_entropy = reduce_mean_valid(curr_entropy)

        self.mean_policy_loss = reduce_mean_valid(-new_surrogate_loss)
        self.mean_vf_loss = tf.constant(0.0)
        loss = reduce_mean_valid(
            -new_surrogate_loss
            # + cur_kl_coeff * action_kl -
            # - entropy_coeff * curr_entropy
        )
        self.loss = loss


# class SACDiversityLoss:
#     def __init__(self, policy, model, cur_obs):
#         model_out_t, _ = model({
#             "obs": cur_obs,
#             "is_training": policy._get_is_training_placeholder(),
#         }, [], None)
#         policy_t, log_pis_t = model.get_policy_output(model_out_t)
#         alpha = model.alpha
#         # Q-values for current policy (no noise) in given current state
#         q_t_det_policy = model.get_q_values(model_out_t, policy_t)
#         assert policy.config["n_step"] == 1
#         actor_loss = tf.reduce_mean(alpha * log_pis_t - q_t_det_policy)
#         self.actor_loss = actor_loss


# def postprocess_dice_sac(policy, sample_batch, others_batches, episode):
#     if not policy.loss_initialized():
#         batch = postprocess_trajectory(policy, sample_batch)
#
#         batch[DIVERSITY_REWARDS] = batch["rewards"].copy()
#         batch[DIVERSITY_VALUE_TARGETS] = batch["rewards"].copy()
#         batch[DIVERSITY_ADVANTAGES] = batch["rewards"].copy()
#         batch['other_action_logp'] = batch[ACTION_LOGP].copy()
#         return batch
#
#     if (not policy.config[PURE_OFF_POLICY]) or (not others_batches):
#         batch = sample_batch.copy()
#         batch = postprocess_trajectory(policy, batch)
#         batch[MY_LOGIT] = batch[BEHAVIOUR_LOGITS]
#         batch = postprocess_diversity(policy, batch, others_batches)
#         batches = [batch]
#     else:
#         batches = []
#
#     for pid, (other_policy, other_batch_raw) in others_batches.items():
#         # other_batch_raw is the data collected by other polices.
#         if policy.config[ONLY_TNB]:
#             break
#         if other_batch_raw is None:
#             continue
#         other_batch_raw = other_batch_raw.copy()
#
#         # Replay this policy to get the action distribution of this policy.
#         replay_result = policy.compute_actions(
#             other_batch_raw[SampleBatch.CUR_OBS]
#         )[2]
#         other_batch_raw[MY_LOGIT] = replay_result[BEHAVIOUR_LOGITS]
#
#         # Compute the diversity reward and diversity advantage of this batch.
#         other_batch_raw = postprocess_diversity(
#             policy, other_batch_raw, others_batches
#         )
#
#         # Compute the task advantage of this batch.
#         batches.append(postprocess_trajectory(policy, other_batch_raw))
#
#     # Merge all batches.
#     batch = SampleBatch.concat_samples(batches) if len(batches) != 1 \
#         else batches[0]
#
#     # del batch.data['new_obs']  # save memory
#     # del batch.data['action_prob']
#     return batch


def dice_sac_loss(policy, model, dist_class, train_batch):
    ret_sac_loss = sac_loss(policy, model, dist_class, train_batch)

    logits, state = model.from_batch(train_batch)
    # action_dist = dist_class(logits, model)

    if state:
        max_seq_len = tf.reduce_max(train_batch["seq_lens"])
        mask = tf.sequence_mask(train_batch["seq_lens"], max_seq_len)
        mask = tf.reshape(mask, [-1])
    else:
        mask = tf.ones_like(
            train_batch["rewards"], dtype=tf.bool
        )

    assert not policy.config[USE_DIVERSITY_VALUE_NETWORK]

    # policy.diversity_loss = SACDiversityLoss()

    # Build the loss for diversity
    # if policy.config[USE_DIVERSITY_VALUE_NETWORK]:
    #     raise NotImplementedError()
    # else:

    # ==========
    # The following is MSE style loss
    policy.diversity_loss = PPOLossTwoSideDiversity(
        # train_batch[DIVERSITY_ADVANTAGES],
        None,
        policy, None,
        # train_batch[SampleBatch.ACTIONS],
        train_batch[SampleBatch.CUR_OBS],
        # policy._log_likelihood,
        # train_batch["action_logp"],
        None,
        mask,
        policy.config["clip_param"],
        policy.policies_pool
        # policy.config["entropy_coeff"]
    ).loss
    policy.diversity_reward_mean = tf.reduce_mean(
        train_batch[SampleBatch.REWARDS]
    )  # FIXME wrong
    # ==========

    # policy.diversity_loss = PPOLossTwoSideDiversity(
    #     train_batch[DIVERSITY_ADVANTAGES],
    #     policy,
    #     train_batch[SampleBatch.ACTIONS],
    #     train_batch[SampleBatch.CUR_OBS],
    #     # policy._log_likelihood,
    #     train_batch["action_logp"],
    #     mask,
    #     policy.config["clip_param"],
    #     policy.policies_pool
    #     # policy.config["entropy_coeff"]
    # ).loss
    # Add the diversity reward as a stat
    # policy.diversity_reward_mean = tf.reduce_mean(
    #     train_batch[DIVERSITY_REWARDS]
    # )
    return ret_sac_loss + policy.diversity_loss


def stats_fn(policy, train_batch):
    ret = {
        "td_error": tf.reduce_mean(policy.td_error),
        "actor_loss": tf.reduce_mean(policy.actor_loss),
        "critic_loss": tf.reduce_mean(policy.critic_loss),
        "mean_q": tf.reduce_mean(policy.q_t),
        "max_q": tf.reduce_max(policy.q_t),
        "min_q": tf.reduce_min(policy.q_t),
        "diversity_total_loss": policy.diversity_loss,
        # "diversity_policy_loss": policy.diversity_loss.mean_policy_loss,
        # "diversity_vf_loss": policy.diversity_loss.mean_vf_loss,
        # "diversity_kl": policy.diversity_loss.mean_kl,
        # "diversity_entropy": policy.diversity_loss.mean_entropy,
        "diversity_reward_mean": policy.diversity_reward_mean,
    }
    return ret


class DiCETargetNetworkMixin:
    """SAC already have target network, so we do not need to maintain
    another set of target network.

    This mixin provide the same API of DiCE:
    policy._compute_clone_network_logits
    """

    def __init__(self):
        @make_tf_callable(self.get_session(), True)
        def compute_clone_network_logits(ob):
            feed_dict = {
                SampleBatch.CUR_OBS: tf.convert_to_tensor(ob),
                # SampleBatch.PREV_ACTIONS: tf.convert_to_tensor(prev_act),
                # SampleBatch.PREV_REWARDS: tf.convert_to_tensor(prev_rew),
                "is_training": tf.convert_to_tensor(False)
            }
            model_out, _ = self.target_model(feed_dict)
            logits = self.target_model.action_model(model_out)
            return logits

        self._compute_clone_network_logits = compute_clone_network_logits


def after_init(policy, obs_space, action_space, config):
    TargetNetworkMixin.__init__(policy, config)
    DiCETargetNetworkMixin.__init__(policy)


def before_loss_init(policy, obs_space, action_space, config):
    DiversityValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    ComputeDiversityMixinModified.__init__(policy)
    ComputeTDErrorMixin.__init__(policy)


def extra_action_fetches_fn(policy):
    ret = {
        BEHAVIOUR_LOGITS: policy.model.action_model(policy.model.last_output())
    }
    if policy.config[USE_DIVERSITY_VALUE_NETWORK]:
        ret[DIVERSITY_VALUES] = policy.model.diversity_value_function()
    return ret


class ComputeDiversityMixinModified(ComputeDiversityMixin):
    def compute_diversity(self, my_batch, others_batches):
        """Compute the diversity of this agent."""
        replays = {}
        if self.config[DELAY_UPDATE]:
            # If in DELAY_UPDATE mode, compute diversity against the target
            # network of each policies.
            for other_name, other_policy in self.policies_pool.items():
                logits = other_policy._compute_clone_network_logits(
                    my_batch[SampleBatch.CUR_OBS],
                    # my_batch[SampleBatch.PREV_ACTIONS],
                    # my_batch[SampleBatch.PREV_REWARDS]
                )
                replays[other_name] = logits
        else:
            raise NotImplementedError()

        # Compute the diversity loss based on the action distribution of
        # this policy and other polices.
        if self.config[DIVERSITY_REWARD_TYPE] == "mse":
            replays = [
                np.split(logit, 2, axis=1)[0] for logit in replays.values()
            ]
            my_act = np.split(my_batch[MY_LOGIT], 2, axis=1)[0]
            return np.mean(
                [
                    (np.square(my_act - other_act)).mean(1)
                    for other_act in replays
                ],
                axis=0
            )
        else:
            raise NotImplementedError()


DiCESACPolicy = SACTFPolicy.with_updates(
    name="DiCESACPolicy",
    get_default_config=lambda: dice_sac_default_config,

    # Finish but not test
    # postprocess_fn=postprocess_dice_sac,
    loss_fn=dice_sac_loss,
    gradients_fn=dice_sac_gradient,
    stats_fn=stats_fn,
    grad_stats_fn=grad_stats_fn,
    before_loss_init=before_loss_init,
    mixins=[
        TargetNetworkMixin, ActorCriticOptimizerMixin,
        ComputeTDErrorMixin, DiCETargetNetworkMixin, DiversityValueNetworkMixin,
        ComputeDiversityMixinModified
    ],
    after_init=after_init,
    extra_action_fetches_fn=extra_action_fetches_fn,
    obs_include_prev_action_reward=True

    # Not checked
    # before_init=setup_early_mixins,
    # before_loss_init=setup_mid_mixins,
    # old_mixins=[
    #     LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
    #     ValueNetworkMixin, DiversityValueNetworkMixin, ComputeDiversityMixin,
    #     TargetNetworkMixin
    # ]
)
