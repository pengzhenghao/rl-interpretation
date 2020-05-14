import gym
from gym.spaces import Box, Discrete
from ray.rllib.agents.ddpg.ddpg_tf_policy import ComputeTDErrorMixin, \
    TargetNetworkMixin
from ray.rllib.agents.sac.sac_tf_policy import SACTFPolicy, \
    postprocess_trajectory
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.tf_ops import make_tf_callable

from toolbox.dice.dice_policy import grad_stats_fn, \
    DiversityValueNetworkMixin, ComputeDiversityMixin
from toolbox.dice.dice_postprocess import MY_LOGIT
from toolbox.dice.dice_sac.dice_sac_config import dice_sac_default_config
from toolbox.dice.dice_sac.dice_sac_gradient import dice_sac_gradient, \
    dice_sac_loss, apply_gradients
from toolbox.dice.dice_sac.dice_sac_model import DiCESACModel
from toolbox.dice.utils import *


# class PPOLossTwoSideDiversity:
#     """Compute the PPO loss for diversity without diversity value network"""
#
#     def __init__(
#             self,
#             advantages,
#             policy,
#             actions,
#             obs,
#             # current_actions_logp,
#             prev_actions_logp,
#             valid_mask,
#             clip_param,
#             policies_pool
#             # entropy_coeff=0.01
#     ):
#         # cur_kl_coeff = 0.0
#
#         def reduce_mean_valid(t):
#             return tf.reduce_mean(tf.boolean_mask(t, valid_mask))
#
#         # prev_dist = dist_class(prev_logits, model)
#
#         # This is important
#         # logp_ratio = tf.exp(curr_action_dist.logp(actions) -
#         # prev_actions_logp)
#         # model_out = policy.model.output
#
#         # A workaround, since the model out is the flatten observation
#
#         # ==========
#         # In this part, I want to compute the diversity by myself
#
#         # my_actions = tf.split(
#         #     policy.model.get_policy_output(obs),
#         #     2, axis=1)[0]
#         #
#         # losses = []
#         # for p in policies_pool.values():
#         #     logit = p.target_model.get_policy_output(obs)
#         #     other_act = tf.split(logit, 2, axis=1)[0]
#         #     losses.append(
#         #         tf.keras.losses.mean_squared_error(other_act, my_actions))
#         # mse_loss = tf.reduce_mean(losses)
#         # self.loss = mse_loss
#         # return
#
#         # ==========
#
#         # This is the PPO style loss
#         model_out = obs
#         distribution_inputs = policy.model.get_policy_output(model_out)
#         action_dist_class = get_dist_class(policy.config, policy.action_space)
#         curr_action_dist = action_dist_class(distribution_inputs,
#         policy.model)
#         current_actions_logp = curr_action_dist.logp(actions)
#         logp_ratio = tf.exp(current_actions_logp - prev_actions_logp)
#         self.debug_ratio = logp_ratio
#         with tf.control_dependencies([tf.check_numerics(logp_ratio,
#                                                         "logp_ratio")]):
#             new_surrogate_loss = advantages * tf.minimum(
#                 logp_ratio, 1 + clip_param
#             )
#
#         # action_kl = prev_dist.kl(curr_action_dist)
#         # self.mean_kl = reduce_mean_valid(action_kl)
#         # curr_entropy = curr_action_dist.entropy()
#         # self.mean_entropy = reduce_mean_valid(curr_entropy)
#
#         self.mean_policy_loss = reduce_mean_valid(-new_surrogate_loss)
#         self.mean_vf_loss = tf.constant(0.0)
#         loss = reduce_mean_valid(
#             -new_surrogate_loss
#             # + cur_kl_coeff * action_kl -
#             # - entropy_coeff * curr_entropy
#         )
#         self.loss = loss


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


# # def postprocess_diversity(policy, batch, others_batches):
#     """Compute the diversity for this policy against other policies using this
#     batch."""
#     # Compute diversity and add a new entry of batch: diversity_rewards
#     batch[DIVERSITY_REWARDS] = policy.compute_diversity(batch, others_batches)
#     assert not np.isscalar(batch[DIVERSITY_REWARDS])
#     return batch


def postprocess_dice_sac(policy, sample_batch, others_batches, episode):
    batches = [postprocess_trajectory(policy, sample_batch.copy())]
    if policy.config[ONLY_TNB] or not policy.loss_initialized():
        batch = batches[0]
        batch["diversity_rewards"] = np.zeros_like(
            batch[SampleBatch.REWARDS], dtype=np.float32)
        return batch
    for pid, (other_policy, other_batch_raw) in others_batches.items():
        # other_batch_raw is the data collected by other polices.
        if other_batch_raw is None:
            continue
        other_batch_raw = other_batch_raw.copy()
        batches.append(postprocess_trajectory(policy, other_batch_raw))
    return SampleBatch.concat_samples(batches) if len(batches) != 1 \
        else batches[0]


def stats_fn(policy, train_batch):
    ret = {
        "td_error": tf.reduce_mean(policy.td_error),
        "actor_loss": tf.reduce_mean(policy.actor_loss),
        "critic_loss": tf.reduce_mean(policy.critic_loss),
        "alpha_loss": tf.reduce_mean(policy.alpha_loss),
        "target_entropy": tf.reduce_mean(policy.target_entropy),
        "entropy": tf.reduce_mean(policy.entropy),
        "mean_q": tf.reduce_mean(policy.q_t),
        "max_q": tf.reduce_max(policy.q_t),
        "min_q": tf.reduce_min(policy.q_t),
        "diversity_actor_loss": tf.reduce_mean(policy.diversity_actor_loss),
        "diversity_critic_loss": tf.reduce_mean(policy.diversity_critic_loss),
        # "diversity_policy_loss": policy.diversity_loss.mean_policy_loss,
        # "diversity_vf_loss": policy.diversity_loss.mean_vf_loss,
        # "diversity_kl": policy.diversity_loss.mean_kl,
        # "diversity_entropy": policy.diversity_loss.mean_entropy,
        "diversity_reward_mean": tf.reduce_mean(policy.diversity_reward_mean),
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
    DiversityValueNetworkMixin.__init__(
        policy, obs_space, action_space, config
    )
    discrete = isinstance(action_space, gym.spaces.Discrete)
    ComputeDiversityMixinModified.__init__(policy, discrete)
    ComputeTDErrorMixin.__init__(policy, dice_sac_loss)


def extra_action_fetches_fn(policy):
    ret = {
        SampleBatch.ACTION_DIST_INPUTS: policy.model.action_model(
            policy.model.last_output()
        )
    }
    # TODO in fact we have the diversity critic
    # if policy.config[USE_DIVERSITY_VALUE_NETWORK]:
    # ret[DIVERSITY_VALUES] = policy.model.get_diversity_q_values(
    #     policy.model.model_out)
    # ret["td_error"] = policy.td_error
    # ret["diversity_td_error"] = policy.diversity_td_error
    # ret["diversity_reward_mean"] = policy.diversity_reward_mean
    return ret


class ComputeDiversityMixinModified(ComputeDiversityMixin):
    def compute_diversity(self, my_batch, policy_map):
        """Compute the diversity of this agent."""
        replays = {}
        assert policy_map
        for other_name, other_policy in policy_map.items():
            logits = other_policy._compute_clone_network_logits(
                my_batch[SampleBatch.CUR_OBS],
            )
            replays[other_name] = logits

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


def build_sac_model(policy, obs_space, action_space, config):
    if config["model"].get("custom_model"):
        logger.warning(
            "Setting use_state_preprocessor=True since a custom model "
            "was specified.")
        config["use_state_preprocessor"] = True
    if not isinstance(action_space, (Box, Discrete)):
        raise UnsupportedSpaceException(
            "Action space {} is not supported for SAC.".format(action_space))
    if isinstance(action_space, Box) and len(action_space.shape) > 1:
        raise UnsupportedSpaceException(
            "Action space has multiple dimensions "
            "{}. ".format(action_space.shape) +
            "Consider reshaping this into a single dimension, "
            "using a Tuple action space, or the multi-agent API.")

    # 2 cases:
    # 1) with separate state-preprocessor (before obs+action concat).
    # 2) no separate state-preprocessor: concat obs+actions right away.
    if config["use_state_preprocessor"]:
        num_outputs = 256  # Flatten last Conv2D to this many nodes.
    else:
        num_outputs = 0
        # No state preprocessor: fcnet_hiddens should be empty.
        if config["model"]["fcnet_hiddens"]:
            logger.warning(
                "When not using a state-preprocessor with SAC, `fcnet_hiddens`"
                " will be set to an empty list! Any hidden layer sizes are "
                "defined via `policy_model.hidden_layer_sizes` and "
                "`Q_model.hidden_layer_sizes`.")
            config["model"]["fcnet_hiddens"] = []

    # Force-ignore any additionally provided hidden layer sizes.
    # Everything should be configured using SAC's "Q_model" and "policy_model"
    # settings.
    policy.model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        framework="torch" if config["use_pytorch"] else "tf",
        model_interface=DiCESACModel,
        name="sac_model",
        actor_hidden_activation=config["policy_model"]["fcnet_activation"],
        actor_hiddens=config["policy_model"]["fcnet_hiddens"],
        critic_hidden_activation=config["Q_model"]["fcnet_activation"],
        critic_hiddens=config["Q_model"]["fcnet_hiddens"],
        twin_q=config["twin_q"],
        diversity_twin_q=config["diversity_twin_q"],
        initial_alpha=config["initial_alpha"],
        target_entropy=config["target_entropy"]
    )

    policy.target_model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        framework="torch" if config["use_pytorch"] else "tf",
        model_interface=DiCESACModel,
        name="target_sac_model",
        actor_hidden_activation=config["policy_model"]["fcnet_activation"],
        actor_hiddens=config["policy_model"]["fcnet_hiddens"],
        critic_hidden_activation=config["Q_model"]["fcnet_activation"],
        critic_hiddens=config["Q_model"]["fcnet_hiddens"],
        twin_q=config["twin_q"],
        diversity_twin_q=config["diversity_twin_q"],
        initial_alpha=config["initial_alpha"],
        target_entropy=config["target_entropy"]
    )

    return policy.model


def extra_learn_fetches_fn(policy):
    ret = dict()
    ret["td_error"] = policy.td_error
    ret["diversity_td_error"] = policy.diversity_td_error
    ret["diversity_reward_mean"] = policy.diversity_reward_mean
    ret["log_pis_t"] = policy.log_pis_t
    return ret


class ActorCriticOptimizerMixin:
    def __init__(self, config):
        # create global step for counting the number of update operations
        self.global_step = tf.train.get_or_create_global_step()

        # use separate optimizers for actor & critic
        self._actor_optimizer = tf.train.AdamOptimizer(
            learning_rate=config["optimization"]["actor_learning_rate"])
        self._critic_optimizer = [
            tf.train.AdamOptimizer(
                learning_rate=config["optimization"]["critic_learning_rate"])
        ]
        self._diversity_critic_optimizer = [
            tf.train.AdamOptimizer(
                learning_rate=config["optimization"]["critic_learning_rate"])
        ]
        if config["twin_q"]:
            self._critic_optimizer.append(
                tf.train.AdamOptimizer(learning_rate=config["optimization"][
                    "critic_learning_rate"]))
        if config["diversity_twin_q"]:
            self._diversity_critic_optimizer.append(
                tf.train.AdamOptimizer(learning_rate=config["optimization"][
                    "critic_learning_rate"]))
        self._alpha_optimizer = tf.train.AdamOptimizer(
            learning_rate=config["optimization"]["entropy_learning_rate"])


def setup_early_mixins(policy, obs_space, action_space, config):
    ActorCriticOptimizerMixin.__init__(policy, config)


DiCESACPolicy = SACTFPolicy.with_updates(
    name="DiCESACPolicy",
    get_default_config=lambda: dice_sac_default_config,

    # Finish but not test
    postprocess_fn=postprocess_dice_sac,
    loss_fn=dice_sac_loss,
    gradients_fn=dice_sac_gradient,
    apply_gradients_fn=apply_gradients,
    stats_fn=stats_fn,
    grad_stats_fn=grad_stats_fn,
    before_loss_init=before_loss_init,
    mixins=[
        TargetNetworkMixin, ActorCriticOptimizerMixin, ComputeTDErrorMixin,
        DiCETargetNetworkMixin, DiversityValueNetworkMixin,
        ComputeDiversityMixinModified
    ],
    before_init=setup_early_mixins,
    after_init=after_init,
    extra_action_fetches_fn=extra_action_fetches_fn,
    extra_learn_fetches_fn=extra_learn_fetches_fn,
    obs_include_prev_action_reward=False,

    # Not checked
    make_model=build_sac_model,

    # before_init=setup_early_mixins,
    # before_loss_init=setup_mid_mixins,
    # old_mixins=[
    #     LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
    #     ValueNetworkMixin, DiversityValueNetworkMixin, ComputeDiversityMixin,
    #     TargetNetworkMixin
    # ]
)
