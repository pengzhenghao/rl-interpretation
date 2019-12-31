import logging

from ray.rllib.agents.ppo.ppo import \
    validate_config as validate_config_original
from ray.rllib.agents.ppo.ppo_policy import postprocess_ppo_gae, ACTION_LOGP, \
    setup_mixins, ValueNetworkMixin, KLCoeffMixin, LearningRateSchedule, \
    EntropyCoeffSchedule, SampleBatch, BEHAVIOUR_LOGITS, make_tf_callable, \
    kl_and_loss_stats, PPOTFPolicy
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.tf_policy import ACTION_PROB
from ray.rllib.utils.explained_variance import explained_variance
from ray.tune.registry import _global_registry, ENV_CREATOR

from toolbox.cooperative_exploration.ceppo_postprocess import \
    postprocess_ppo_gae_replay
from toolbox.dece.dece_loss import loss_dece, tnb_gradients
from toolbox.dece.utils import *
from toolbox.distance import get_kl_divergence
from toolbox.ppo_es.tnb_es import TNBESTrainer

# from toolbox.ipd.tnb_utils import *

logger = logging.getLogger(__name__)


def _compute_logp(logit, x):
    # Only for DiagGaussian distribution. Copied from tf_action_dist.py
    logit = logit.astype(np.float64)
    x = np.expand_dims(x.astype(np.float64), 1) if x.ndim == 1 else x
    mean, log_std = np.split(logit, 2, axis=1)
    logp = (
            -0.5 * np.sum(np.square((x - mean) / np.exp(log_std)), axis=1) -
            0.5 * np.log(2.0 * np.pi) * x.shape[1] - np.sum(log_std, axis=1)
    )
    p = np.exp(logp)
    return logp, p


def _clip_batch(other_batch, clip_action_prob_kl):
    kl = get_kl_divergence(
        source=other_batch[BEHAVIOUR_LOGITS],
        target=other_batch["other_logits"],
        mean=False
    )

    mask = kl < clip_action_prob_kl
    length = len(mask)
    info = {"kl": kl, "unclip_length": length, "length": len(mask)}

    if not np.all(mask):
        length = mask.argmin()
        info['unclip_length'] = length
        if length == 0:
            return None, info
        assert length < len(other_batch['action_logp'])
        other_batch = other_batch.slice(0, length)

    return other_batch, info


def postprocess_dece(policy, sample_batch, others_batches=None, episode=None):
    if not policy.loss_initialized():
        batch = postprocess_ppo_gae(policy, sample_batch)
        batch["advantages_unnormalized"] = np.zeros_like(
            batch["advantages"], dtype=np.float32
        )
        batch['debug_ratio'] = np.zeros_like(
            batch["advantages"], dtype=np.float32
        )
        batch['debug_fake_adv'] = np.zeros_like(
            batch["advantages"], dtype=np.float32
        )
        # if policy.config[DIVERSITY_ENCOURAGING] or policy.config[CURIOSITY]:
        #     # assert not policy.config["use_joint_dataset"]
        #     batch[JOINT_OBS] = np.zeros_like(
        #         sample_batch[SampleBatch.CUR_OBS], dtype=np.float32
        #     )
        #     batch[PEER_ACTION] = np.zeros_like(
        #         sample_batch[SampleBatch.ACTIONS], dtype=np.float32
        #     )
        return batch

    batch = sample_batch

    if policy.config[REPLAY_VALUES]:
        # a little workaround. We normalize advantage for all batch before
        # concatnation.
        tmp_batch = postprocess_ppo_gae(policy, batch)
        value = tmp_batch[Postprocessing.ADVANTAGES]
        standardized = (value - value.mean()) / max(1e-4, value.std())
        tmp_batch[Postprocessing.ADVANTAGES] = standardized
        batches = [tmp_batch]
    else:
        batches = [postprocess_ppo_gae(policy, batch)]

    for pid, (other_policy, other_batch_raw) in others_batches.items():
        # The logic is that EVEN though we may use DISABLE or NO_REPLAY_VALUES,
        # but we still want to take a look of those statics.
        # Maybe in the future we can add knob to remove all such slowly stats.

        if other_batch_raw is None:
            continue

        other_batch = other_batch_raw.copy()

        # four fields that we will overwrite.
        # Two additional: advantages / value target
        other_batch["other_action_logp"] = other_batch[ACTION_LOGP].copy()
        other_batch["other_action_prob"] = other_batch[ACTION_PROB].copy()
        other_batch["other_logits"] = other_batch[BEHAVIOUR_LOGITS].copy()
        other_batch["other_vf_preds"] = other_batch[SampleBatch.VF_PREDS
        ].copy()

        # use my policy to evaluate the values and other relative data
        # of other's samples.
        replay_result = policy.compute_actions(
            other_batch[SampleBatch.CUR_OBS]
        )[2]

        other_batch[SampleBatch.VF_PREDS] = replay_result[SampleBatch.VF_PREDS]
        other_batch[BEHAVIOUR_LOGITS] = replay_result[BEHAVIOUR_LOGITS]

        other_batch[ACTION_LOGP], other_batch[ACTION_PROB] = \
            _compute_logp(
                other_batch[BEHAVIOUR_LOGITS],
                other_batch[SampleBatch.ACTIONS]
            )

        if policy.config[DISABLE]:
            continue
        elif not policy.config[REPLAY_VALUES]:
            batches.append(postprocess_ppo_gae(policy, other_batch_raw))
        else:  # replay values
            if other_batch is not None:  # it could be None due to clipping.
                batches.append(
                    postprocess_ppo_gae_replay(
                        policy, other_batch, other_policy
                    )
                )

    for batch in batches:
        batch[Postprocessing.ADVANTAGES + "_unnormalized"] = batch[
            Postprocessing.ADVANTAGES].copy().astype(np.float32)
        if "debug_ratio" not in batch:
            assert "debug_fake_adv" not in batch
            batch['debug_fake_adv'] = batch['debug_ratio'] = np.zeros_like(
                batch['advantages'], dtype=np.float32
            )

    return SampleBatch.concat_samples(batches) if len(batches) != 1 \
        else batches[0]


def setup_mixins_ceppo(policy, obs_space, action_space, config):
    setup_mixins(policy, obs_space, action_space, config)


def wrap_stats_ceppo(policy, train_batch):
    # if policy.config[DIVERSITY_ENCOURAGING]:
    #     return wrap_stats_fn(policy, train_batch)
    ret = kl_and_loss_stats(policy, train_batch)
    if hasattr(policy.loss_obj, "stats"):
        assert isinstance(policy.loss_obj.stats, dict)
        ret.update(policy.loss_obj.stats)
    return ret


def grad_stats_fn(policy, batch, grads):
    if not policy.enable_novelty:
        return {}
    ret = {
        "cos_similarity": policy.gradient_cosine_similarity,
        "policy_grad_norm": policy.policy_grad_norm,
        "novelty_grad_norm": policy.novelty_grad_norm
    }
    return ret


class NoveltyValueNetworkMixin(object):
    def __init__(self, obs_space, action_space, config):
        if config["use_gae"] and config['use_novelty_value_network']:

            @make_tf_callable(self.get_session())
            def novelty_value(ob, prev_action, prev_reward, *state):
                model_out, _ = self.model(
                    {
                        SampleBatch.CUR_OBS: tf.convert_to_tensor([ob]),
                        SampleBatch.PREV_ACTIONS: tf.convert_to_tensor(
                            [prev_action]
                        ),
                        SampleBatch.PREV_REWARDS: tf.convert_to_tensor(
                            [prev_reward]
                        ),
                        "is_training": tf.convert_to_tensor(False),
                    }, [tf.convert_to_tensor([s]) for s in state],
                    tf.convert_to_tensor([1])
                )
                return self.model.novelty_value_function()[0]

        else:

            @make_tf_callable(self.get_session())
            def novelty_value(ob, prev_action, prev_reward, *state):
                return tf.constant(0.0)

        self._novelty_value = novelty_value


def additional_fetches(policy):
    """Adds value function and logits outputs to experience train_batches."""
    ret = {
        SampleBatch.VF_PREDS: policy.model.value_function(),
        BEHAVIOUR_LOGITS: policy.model.last_output()
    }
    if policy.config['use_novelty_value_network']:
        ret[NOVELTY_VALUES] = policy.model.novelty_value_function()
    return ret


def kl_and_loss_stats_modified(policy, train_batch):
    ret = kl_and_loss_stats(policy, train_batch)
    if not policy.enable_novelty:
        return ret
    ret.update(
        {
            "novelty_total_loss": policy.novelty_loss_obj.loss,
            "novelty_policy_loss": policy.novelty_loss_obj.mean_policy_loss,
            "novelty_vf_loss": policy.novelty_loss_obj.mean_vf_loss,
            "novelty_kl": policy.novelty_loss_obj.mean_kl,
            "novelty_entropy": policy.novelty_loss_obj.mean_entropy,
            "novelty_reward_mean": policy.novelty_reward_mean,
            "novelty_reward_ratio": policy.novelty_reward_ratio
        }
    )
    if policy.config['use_novelty_value_network']:
        ret['novelty_vf_explained_var'] = explained_variance(
            train_batch[NOVELTY_VALUE_TARGETS],
            policy.model.novelty_value_function()
        )
    return ret


class ComputeNoveltyMixin(object):

    def __init__(self):
        self.enable_novelty = True

    def compute_novelty(self, my_batch, others_batches, episode):
        if not others_batches:
            return np.zeros_like(my_batch[SampleBatch.REWARDS],
                                 dtype=np.float32)

        replays = []
        for (other_policy, _) in others_batches.values():
            _, _, info = other_policy.compute_actions(
                my_batch[SampleBatch.CUR_OBS])
            replays.append(info[BEHAVIOUR_LOGITS])

        assert replays

        if self.config["novelty_type"] == "kl":
            return np.mean(
                [get_kl_divergence(
                    my_batch[BEHAVIOUR_LOGITS], logit, mean=False
                ) for logit in replays
                ], axis=0)

        elif self.config["novelty_type"] == "mse":
            replays = [np.split(logit, 2, axis=1)[0] for logit in replays]
            my_act = np.split(my_batch[BEHAVIOUR_LOGITS], 2, axis=1)[0]
            return np.mean(
                [(np.square(my_act - other_act)).mean(1) for other_act in
                 replays], axis=0)
        else:
            raise NotImplementedError()


def setup_mixins_tnb(policy, action_space, obs_space, config):
    setup_mixins(policy, action_space, obs_space, config)
    NoveltyValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    ComputeNoveltyMixin.__init__(policy)


DECEPolicy = PPOTFPolicy.with_updates(
    name="DECEPolicy",
    get_default_config=lambda: dece_default_config,
    postprocess_fn=postprocess_dece,
    loss_fn=loss_dece,

    stats_fn=kl_and_loss_stats_modified,
    gradients_fn=tnb_gradients,
    grad_stats_fn=grad_stats_fn,
    extra_action_fetches_fn=additional_fetches,

    before_loss_init=setup_mixins_tnb,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin, NoveltyValueNetworkMixin, ComputeNoveltyMixin
    ]
)


# FIXME the key modification is the loss. We introduce new element such as
#  the 'other_values' in postprocess. Then we generate advantage based on them.
#  We then use the


def validate_config(config):
    # create multi-agent environment
    assert _global_registry.contains(ENV_CREATOR, config["env"])
    env_creator = _global_registry.get(ENV_CREATOR, config["env"])
    tmp_env = env_creator(config["env_config"])
    config["multiagent"]["policies"] = {
        i: (None, tmp_env.observation_space, tmp_env.action_space, {})
        for i in tmp_env.agent_ids
    }
    config["multiagent"]["policy_mapping_fn"] = lambda x: x

    # check the model
    if config[USE_BISECTOR]:
        assert config['model']['custom_model'] == "ActorDoubleCriticNetwork"
        config['model']['custom_options'] = {
            "use_novelty_value_network": config['use_novelty_value_network']
        }

    # Reduce the train batch size for each agent
    num_agents = len(config['multiagent']['policies'])
    config['train_batch_size'] = int(
        config['train_batch_size'] // num_agents
    )
    assert config['train_batch_size'] >= config["sgd_minibatch_size"]

    validate_config_original(config)


DECETrainer = TNBESTrainer.with_updates(
    name="DECETrainer",
    default_config=dece_default_config,
    default_policy=DECEPolicy,
    validate_config=validate_config
)
# FIXME So till now the only change to TNBESTrainer
