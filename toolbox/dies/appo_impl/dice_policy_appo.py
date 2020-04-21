"""
This file implement a DiCE policy. Note that in one DiCE trainer, there are
many DiCE policies, each serves as a member in the team. We implement the
following functions for each policy:
1. Compute the diversity of one policy against others.
2. Maintain the target network for each policy if in DELAY_UPDATE mode.
3. Update the target network for each training iteration.
"""
import logging

import numpy as np
from ray.rllib.agents.ppo.appo_policy import AsyncPPOTFPolicy, KLCoeffMixin, \
    ValueNetworkMixin, \
    setup_mixins as original_setup_mixins, \
    add_values_and_logits as original_additional_fetch, stats as original_stats

from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import LearningRateSchedule
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.explained_variance import explained_variance
from ray.rllib.utils.tf_ops import make_tf_callable

from toolbox.dies.appo_impl.constants import *
from toolbox.dies.appo_impl.dice_loss_appo import build_appo_surrogate_loss, \
    dice_gradient, BEHAVIOUR_LOGITS
from toolbox.dies.appo_impl.dice_postprocess_appo import postprocess_dice, \
    MY_LOGIT
from toolbox.distance import get_kl_divergence

tf = try_import_tf()

logger = logging.getLogger(__name__)


def grad_stats_fn(policy, batch, grads):
    if policy.config.get(I_AM_CLONE, False):
        return {}

    if policy.config[USE_BISECTOR]:
        ret = {
            "cos_similarity": policy.gradient_cosine_similarity,
            "policy_grad_norm": policy.policy_grad_norm,
            "diversity_grad_norm": policy.diversity_grad_norm
        }
        return ret
    else:
        return {}


class DiversityValueNetworkMixin:
    def __init__(self, obs_space, action_space, config):
        if config["use_gae"] and config[USE_DIVERSITY_VALUE_NETWORK]:

            @make_tf_callable(self.get_session())
            def diversity_value(ob, prev_action, prev_reward, *state):
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
                return self.model.diversity_value_function()[0]
        else:

            @make_tf_callable(self.get_session())
            def diversity_value(ob, prev_action, prev_reward, *state):
                return tf.constant(0.0)

        self._diversity_value = diversity_value


def additional_fetches(policy):
    """Fetch diversity values if using diversity value network."""
    ret = original_additional_fetch(policy)
    if policy.config[USE_DIVERSITY_VALUE_NETWORK]:
        ret[DIVERSITY_VALUES] = policy.model.diversity_value_function()
    return ret


def kl_and_loss_stats_modified(policy, train_batch):
    """Add the diversity-related stats here."""

    if policy.config.get(I_AM_CLONE, False):
        return {}

    ret = original_stats(policy, train_batch)
    ret.update({
        "diversity_total_loss": policy.diversity_loss.total_loss,
        "diversity_policy_loss": policy.diversity_loss.pi_loss,
        "diversity_kl": policy.diversity_loss.mean_kl,
        "diversity_entropy": policy.diversity_loss.entropy,
        "diversity_reward_mean": policy.diversity_reward_mean,  # ?
    })
    if policy.config[USE_DIVERSITY_VALUE_NETWORK]:
        ret['diversity_vf_explained_var'] = explained_variance(
            train_batch[DIVERSITY_VALUE_TARGETS],
            policy.model.diversity_value_function()
        )
        ret["diversity_vf_loss"] = policy.diversity_loss.vf_loss
    return ret


class ComputeDiversityMixin:
    """This class initialize a reference of the policies pool within each
    policy, and provide the function to compute the diversity of each policy.

    The _lazy_initialize is only called in DELAY_UPDATE mode. This is because
    if we compute diversity of this policy against other latest policies,
    we can simply access other policies via other_batches, the input to the
    compute_diversity function.
    """

    def __init__(self):
        self.initialized_policies_pool = False
        self.policy_pool = {}

    def _lazy_initialize(self, policies_pool):
        """Initialize the reference of policies pool within this policy."""
        self.policy_pool = policies_pool
        self.num_of_policies = len(self.policy_pool)
        self.initialized_policies_pool = True

    def compute_diversity(self, my_batch):
        """Compute the diversity of this agent."""
        assert self.policy_pool, "Your policies pool is empty!"
        replays = {}
        for other_name, other_policy in self.policy_pool.items():
            _, _, info = other_policy.compute_actions(
                my_batch[SampleBatch.CUR_OBS]
            )
            replays[other_name] = info[BEHAVIOUR_LOGITS]

        # Compute the diversity loss based on the action distribution of
        # this policy and other polices.
        assert replays
        if self.config[DIVERSITY_REWARD_TYPE] == "kl":
            return np.mean(
                [
                    get_kl_divergence(my_batch[MY_LOGIT], logit, mean=False)
                    for logit in replays.values()
                ],
                axis=0
            )

        elif self.config[DIVERSITY_REWARD_TYPE] == "mse":
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



def setup_mixins_dice(policy, action_space, obs_space, config):
    original_setup_mixins(policy, action_space, obs_space, config)
    DiversityValueNetworkMixin.__init__(policy, obs_space, action_space,
                                        config)
    ComputeDiversityMixin.__init__(policy)


def setup_late_mixins(policy, obs_space, action_space, config):
    # If you use Vtrace, then you should set policy's after_init to this func.
    # if config[DELAY_UPDATE]:
    #     TargetNetworkMixin.__init__(policy, obs_space, action_space, config)
    pass


DiCEPolicy_APPO = AsyncPPOTFPolicy.with_updates(
    name="DiCEPolicy_APPO",
    get_default_config=lambda: dice_appo_default_config,
    postprocess_fn=postprocess_dice,
    loss_fn=build_appo_surrogate_loss,
    stats_fn=kl_and_loss_stats_modified,
    gradients_fn=dice_gradient,
    grad_stats_fn=grad_stats_fn,
    extra_action_fetches_fn=additional_fetches,
    before_loss_init=setup_mixins_dice,
    mixins=[
        LearningRateSchedule,
        # EntropyCoeffSchedule,
        KLCoeffMixin,
        # TargetNetworkMixin,
        ValueNetworkMixin,
        DiversityValueNetworkMixin,
        ComputeDiversityMixin,

    ]
)
