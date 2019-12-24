import logging

import numpy as np
import tensorflow as tf
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG, PPOTrainer, PPOTFPolicy
from ray.rllib.agents.ppo.ppo_policy import setup_mixins, ValueNetworkMixin, \
    KLCoeffMixin, LearningRateSchedule, EntropyCoeffSchedule, \
    SampleBatch, BEHAVIOUR_LOGITS, make_tf_callable, PPOLoss
from ray.rllib.evaluation.postprocessing import Postprocessing, discount
from ray.rllib.models import ModelCatalog
from ray.tune.util import merge_dicts
from ray.rllib.utils.explained_variance import explained_variance

from toolbox.ipd.tnb_model import ActorDoubleCriticNetwork

logger = logging.getLogger(__name__)

ipd_default_config = merge_dicts(
    DEFAULT_CONFIG,
    dict(
        model={"custom_model": "ActorDoubleCriticNetwork"},
        novelty_threshold=0.5,
        distance_mode="min",
        use_second_component=True,
        clip_novelty_gradient=False
    )
)

NOVELTY_REWARDS = "novelty_rewards"
NOVELTY_VALUES = "novelty_values"
NOVELTY_ADVANTAGES = "novelty_advantages"
NOVELTY_VALUE_TARGETS = "novelty_value_targets"

ModelCatalog.register_custom_model(
    "ActorDoubleCriticNetwork", ActorDoubleCriticNetwork
)


def get_action_mean(logits):
    return np.split(logits, 2, axis=1)[0]


def postprocess_ipd(policy, sample_batch, other_batches, episode):
    completed = sample_batch["dones"][-1]
    sample_batch[NOVELTY_REWARDS] = policy.compute_novelty(
        sample_batch[SampleBatch.CUR_OBS], sample_batch[SampleBatch.ACTIONS]
    )

    if completed:
        last_r_novelty = last_r = 0.0
    else:
        next_state = []
        for i in range(policy.num_state_tensors()):
            next_state.append([sample_batch["state_out_{}".format(i)][-1]])
        last_r = policy._value(sample_batch[SampleBatch.NEXT_OBS][-1],
                               sample_batch[SampleBatch.ACTIONS][-1],
                               sample_batch[SampleBatch.REWARDS][-1],
                               *next_state)
        last_r_novelty = policy._novelty_value(
            sample_batch[SampleBatch.NEXT_OBS][-1],
            sample_batch[SampleBatch.ACTIONS][-1],
            sample_batch[NOVELTY_REWARDS][-1],
            *next_state
        )

    # compute the advantages of original rewards
    advantages, value_target = compute_advantages(
        sample_batch[SampleBatch.REWARDS],
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        sample_batch[SampleBatch.VF_PREDS],
        use_gae=policy.config["use_gae"]
    )
    sample_batch[Postprocessing.ADVANTAGES] = advantages
    sample_batch[Postprocessing.VALUE_TARGETS] = value_target

    # compute the advantages of novelty rewards
    novelty_advantages, novelty_value_target = compute_advantages(
        sample_batch[NOVELTY_REWARDS],
        last_r_novelty,
        policy.config["gamma"],
        policy.config["lambda"],
        sample_batch[NOVELTY_VALUES],
        use_gae=policy.config["use_gae"]
    )
    sample_batch[NOVELTY_ADVANTAGES] = novelty_advantages
    sample_batch[NOVELTY_VALUE_TARGETS] = novelty_value_target

    return sample_batch


def compute_advantages(rewards, last_r, gamma, lambda_, values, use_gae=True):
    if use_gae:
        vpred_t = np.concatenate([values, np.array([last_r])])
        delta_t = (rewards + gamma * vpred_t[1:] - vpred_t[:-1])
        advantage = discount(delta_t, gamma * lambda_)
        value_target = (advantage + values).copy().astype(np.float32)
    else:
        rewards_plus_v = np.concatenate([rewards, np.array([last_r])])
        advantage = discount(rewards_plus_v, gamma)[:-1]
        # TODO(ekl): support using a critic without GAE
        value_target = np.zeros_like(advantage)
    advantage = advantage.copy().astype(np.float32)
    return advantage, value_target


class AgentPoolMixin(object):
    def __init__(self, path_list, threshold, distance_mode='min'):
        # restore a buffer of policies from disk.
        self.policies_pool = []

        assert not path_list
        for path in path_list:
            # TODO restore agent. add check.
            policy = None
            policy_info = {}
            self.policies_pool.append(dict(
                policy=policy,
                **policy_info
            ))

        # self.Policy_Buffer = Policy_Buffer
        self.num_of_policies = len(self.policies_pool)

        self.novelty_recorder = np.zeros(self.num_of_policies)
        self.novelty_recorder_len = 0

        self.threshold = threshold
        self.distance_mode = distance_mode
        assert distance_mode in ['min', 'max']

    def compute_novelty(self, state, action):
        if len(self.policies_pool) == 0:
            return np.zeros((action.shape[0],), dtype=np.float32)
        for i, (key, policy_dict) in enumerate(self.policies_pool.items()):
            policy = policy_dict['policy']
            _, _, info = policy.compute_actions(state)
            other_action = get_action_mean(info[BEHAVIOUR_LOGITS])
            self.novelty_recorder[i] += np.linalg.norm(other_action - action)
        self.novelty_recorder_len += 1
        if self.distance_mode == 'min':
            min_novel = np.min(
                self.novelty_recorder / self.novelty_recorder_len)
            return min_novel - self.threshold
        elif self.distance_mode == 'max':
            max_novel = np.max(
                self.novelty_recorder / self.novelty_recorder_len)
            return max_novel - self.threshold
        else:
            raise NotImplementedError()


class NoveltyValueNetworkMixin(object):
    def __init__(self, obs_space, action_space, config):
        if config["use_gae"]:

            @make_tf_callable(self.get_session())
            def novelty_value(ob, prev_action, prev_reward, *state):
                model_out, _ = self.model({
                    SampleBatch.CUR_OBS: tf.convert_to_tensor([ob]),
                    SampleBatch.PREV_ACTIONS: tf.convert_to_tensor(
                        [prev_action]),
                    SampleBatch.PREV_REWARDS: tf.convert_to_tensor(
                        [prev_reward]),
                    "is_training": tf.convert_to_tensor(False),
                }, [tf.convert_to_tensor([s]) for s in state],
                    tf.convert_to_tensor([1]))
                return self.model.novelty_value_function()[0]

        else:

            @make_tf_callable(self.get_session())
            def novelty_value(ob, prev_action, prev_reward, *state):
                return tf.constant(0.0)

        self._novelty_value = novelty_value


def setup_mixins_tnb(policy, action_space, obs_space, config):
    setup_mixins(policy, action_space, obs_space, config)
    NoveltyValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    AgentPoolMixin.__init__(policy, [], config['novelty_threshold'],
                            config['distance_mode'])


def additional_fetches(policy):
    """Adds value function and logits outputs to experience train_batches."""
    return {
        SampleBatch.VF_PREDS: policy.model.value_function(),
        BEHAVIOUR_LOGITS: policy.model.last_output(),
        NOVELTY_VALUES: policy.model.novelty_value_function()
    }


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
            train_batch[Postprocessing.ADVANTAGES], dtype=tf.bool)

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
        model_config=policy.config["model"])

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
        model_config=policy.config["model"])

    return [policy.loss_obj.loss, policy.novelty_loss_obj.loss]


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
    # policy_grad_flatten = []
    # policy_grad_info = []
    # for idx, (pg, var) in enumerate(policy_grad):
    #     if novelty_grad[idx][0] is None:
    #         # Some variables do not related to novelty, so the grad is None
    #         policy_grad_info.append((None, None, var))
    #         continue
    #
    #     if pg is None:
    #         continue
    #     pg_flat, pg_shape, pg_flat_shape = _flatten(pg)
    #     policy_grad_flatten.append(pg_flat)
    #     policy_grad_info.append((pg_flat_shape, pg_shape, var))
    # if len(policy_grad_flatten) < 2:
    #     raise ValueError(
    #         err_msg.format("policy_grad_flatten", len(policy_grad_flatten))
    #     )
    # # flatten the novelty grad
    # novelty_grad_flatten = []
    # novelty_grad_info = []
    # for ng, _ in novelty_grad:
    #     if ng is None:
    #         novelty_grad_info.append((None, None))
    #         continue
    #     pg_flat, pg_shape, pg_flat_shape = _flatten(ng)
    #     novelty_grad_flatten.append(pg_flat)
    #     novelty_grad_info.append((pg_flat_shape, pg_shape))
    # if len(novelty_grad_flatten) < 2:
    #     raise ValueError(
    #         err_msg.format("novelty_grad_flatten", len(novelty_grad_flatten))
    #     )

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
    policy.novelty_grad_norm = tf.norm(novelty_grad_norm)

    if policy.config["use_second_component"]:
        total_grad = tf.cond(cos_similarity > 0, less_90_deg, greater_90_deg)
    else:
        total_grad = less_90_deg()

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


def grad_stats_fn(policy, batch, grads):
    ret = {
        "cos_similarity": policy.gradient_cosine_similarity,
        "policy_grad_norm": policy.policy_grad_norm,
        "novelty_grad_norm": policy.novelty_grad_norm
    }
    return ret


def kl_and_loss_stats(policy, train_batch):
    return {
        "cur_kl_coeff": tf.cast(policy.kl_coeff, tf.float64),
        "cur_lr": tf.cast(policy.cur_lr, tf.float64),
        "total_loss": policy.loss_obj.loss,
        "novelty_total_loss": policy.novelty_loss_obj.loss,
        "policy_loss": policy.loss_obj.mean_policy_loss,
        "novelty_policy_loss": policy.novelty_loss_obj.mean_policy_loss,
        "vf_loss": policy.loss_obj.mean_vf_loss,
        "novelty_vf_loss": policy.novelty_loss_obj.mean_vf_loss,
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS],
            policy.model.value_function()),
        "novelty_vf_explained_var": explained_variance(
            train_batch[NOVELTY_VALUE_TARGETS],
            policy.model.novelty_value_function()),
        "kl": policy.loss_obj.mean_kl,
        "novelty_kl": policy.novelty_loss_obj.mean_kl,
        "entropy": policy.loss_obj.mean_entropy,
        "novelty_entropy": policy.novelty_loss_obj.mean_entropy,
        "entropy_coeff": tf.cast(policy.entropy_coeff, tf.float64),
    }


TNBPolicy = PPOTFPolicy.with_updates(
    name="TNBPolicy",
    get_default_config=lambda: ipd_default_config,
    before_loss_init=setup_mixins_tnb,
    extra_action_fetches_fn=additional_fetches,
    postprocess_fn=postprocess_ipd,
    loss_fn=tnb_loss,
    stats_fn=kl_and_loss_stats,
    gradients_fn=tnb_gradients,
    grad_stats_fn=grad_stats_fn,
    mixins=[LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
            ValueNetworkMixin, NoveltyValueNetworkMixin, AgentPoolMixin]
)

TNBTrainer = PPOTrainer.with_updates(
    name="TNBPPO",
    default_config=ipd_default_config,
    default_policy=TNBPolicy
)
