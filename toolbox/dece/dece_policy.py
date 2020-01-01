import logging

from ray.rllib.agents.ppo.ppo_policy import setup_mixins, ValueNetworkMixin, KLCoeffMixin, LearningRateSchedule, EntropyCoeffSchedule, SampleBatch, BEHAVIOUR_LOGITS, make_tf_callable, kl_and_loss_stats, PPOTFPolicy
from ray.rllib.utils.explained_variance import explained_variance

from toolbox.dece.dece_loss import loss_dece, tnb_gradients
from toolbox.dece.dece_postprocess import postprocess_dece
from toolbox.dece.utils import *
from toolbox.distance import get_kl_divergence

logger = logging.getLogger(__name__)


def wrap_stats_ceppo(policy, train_batch):
    ret = kl_and_loss_stats(policy, train_batch)
    if hasattr(policy.loss_obj, "stats"):
        assert isinstance(policy.loss_obj.stats, dict)
        ret.update(policy.loss_obj.stats)
    return ret


def grad_stats_fn(policy, batch, grads):
    if policy.config[USE_BISECTOR]:
        ret = {
            "cos_similarity": policy.gradient_cosine_similarity,
            "policy_grad_norm": policy.policy_grad_norm,
            "novelty_grad_norm": policy.novelty_grad_norm
        }
        return ret
    else:
        return {}


class NoveltyValueNetworkMixin(object):
    def __init__(self, obs_space, action_space, config):
        if config["use_gae"] and config[USE_DIVERSITY_VALUE_NETWORK]:

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
    if policy.config[USE_DIVERSITY_VALUE_NETWORK]:
        ret[NOVELTY_VALUES] = policy.model.novelty_value_function()
    return ret


def kl_and_loss_stats_modified(policy, train_batch):
    ret = kl_and_loss_stats(policy, train_batch)
    if not policy.config[DIVERSITY_ENCOURAGING]:
        return ret
    ret.update(
        {
            "novelty_total_loss": policy.novelty_loss_obj.loss,
            "novelty_policy_loss": policy.novelty_loss_obj.mean_policy_loss,
            "novelty_vf_loss": policy.novelty_loss_obj.mean_vf_loss,
            "novelty_kl": policy.novelty_loss_obj.mean_kl,
            "novelty_entropy": policy.novelty_loss_obj.mean_entropy,
            "novelty_reward_mean": policy.novelty_reward_mean
        }
    )
    if policy.config[USE_DIVERSITY_VALUE_NETWORK]:
        ret['novelty_vf_explained_var'] = explained_variance(
            train_batch[NOVELTY_VALUE_TARGETS],
            policy.model.novelty_value_function()
        )
    return ret


class ComputeNoveltyMixin(object):
    def __init__(self):
        self.initialized_policies_pool = False
        self.policies_pool = {}

    def _lazy_initialize(self, policies_pool, my_name):
        assert self.config[DELAY_UPDATE]
        self.policies_pool = {
            agent_name: other_policy
            for agent_name, other_policy in policies_pool.items()
            if agent_name != my_name
        }
        self.num_of_policies = len(self.policies_pool)
        self.initialized_policies_pool = True

    def compute_novelty(self, my_batch, others_batches, episode=None):
        """It should be noted that in Cooperative Exploration setting,
        This implementation is inefficient. Since the 'observation' of each
        agent are identical, though may different in order, so we call the
        compute_actions for num_agents * num_agents * batch_size times overall.
        """
        if not others_batches:
            return np.zeros_like(
                my_batch[SampleBatch.REWARDS], dtype=np.float32
            )
        replays = {}
        if self.config[DELAY_UPDATE]:
            policies_dict = self.policies_pool
        else:
            policies_dict = {k: p for k, (p, _) in others_batches.items()}
        for other_name, other_policy in policies_dict.items():
            _, _, info = other_policy.compute_actions(
                my_batch[SampleBatch.CUR_OBS]
            )
            replays[other_name] = info[BEHAVIOUR_LOGITS]
        if self.config[DIVERSITY_REWARD_TYPE] == "kl":
            return np.mean(
                [
                    get_kl_divergence(
                        my_batch[BEHAVIOUR_LOGITS], logit, mean=False
                    ) for logit in replays.values()
                ],
                axis=0
            )

        elif self.config[DIVERSITY_REWARD_TYPE] == "mse":
            replays = [
                np.split(logit, 2, axis=1)[0] for logit in replays.values()
            ]
            my_act = np.split(my_batch[BEHAVIOUR_LOGITS], 2, axis=1)[0]
            return np.mean(
                [
                    (np.square(my_act - other_act)).mean(1)
                    for other_act in replays
                ],
                axis=0
            )
        else:
            raise NotImplementedError()


def setup_mixins_dece(policy, action_space, obs_space, config):
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
    before_loss_init=setup_mixins_dece,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin, NoveltyValueNetworkMixin, ComputeNoveltyMixin
    ]
)
