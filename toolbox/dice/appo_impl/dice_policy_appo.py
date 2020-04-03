"""
This file implement a DiCE policy. Note that in one DiCE trainer, there are
many DiCE policies, each serves as a member in the team. We implement the
following functions for each policy:
1. Compute the diversity of one policy against others.
2. Maintain the target network for each policy if in DELAY_UPDATE mode.
3. Update the target network for each training iteration.
"""
from ray.rllib.agents.ppo.appo_policy import AsyncPPOTFPolicy, KLCoeffMixin, \
    ValueNetworkMixin, \
    setup_mixins as original_setup_mixins, \
    add_values_and_logits as original_additional_fetch, stats as original_stats
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import LearningRateSchedule
from ray.rllib.utils.explained_variance import explained_variance
from ray.rllib.utils.tf_ops import make_tf_callable

from toolbox.dice.appo_impl.dice_loss_appo import build_appo_surrogate_loss, \
    dice_gradient, BEHAVIOUR_LOGITS
from toolbox.dice.appo_impl.dice_postprocess_appo import postprocess_dice, \
    MY_LOGIT
from toolbox.dice.appo_impl.utils import dice_appo_default_config
from toolbox.dice.utils import *

logger = logging.getLogger(__name__)


def grad_stats_fn(policy, batch, grads):
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
        # assert self.config[DELAY_UPDATE]
        self.policy_pool = policies_pool
        # {
        #     agent_name: other_policy
        #     for agent_name, other_policy in policies_pool.items()
        #     if agent_name != my_name
        # }  # Since it must in DELAY_UPDATE mode, we allow reuse all polices.
        self.num_of_policies = len(self.policy_pool)
        self.initialized_policies_pool = True

    def compute_diversity(self, my_batch, others_batches):
        """Compute the diversity of this agent."""
        assert self.policy_pool, "Your policies pool is empty!"
        replays = {}
        if self.config[DELAY_UPDATE]:
            # If in DELAY_UPDATE mode, compute diversity against the target
            # network of each policies.
            for other_name, other_policy in self.policy_pool.items():
                logits = other_policy._compute_clone_network_logits(
                    my_batch[SampleBatch.CUR_OBS]
                )
                replays[other_name] = logits
        else:
            # Otherwise compute the diversity against other latest policies
            # contained in other_batches.
            if not others_batches:
                return np.zeros_like(
                    my_batch[SampleBatch.REWARDS], dtype=np.float32
                )
            for other_name, (other_policy, _) in others_batches.items():
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


class TargetNetworkMixin:
    """This class implement the DELAY_UPDATE mechanism. Allowing:
    1. delayed update the targets networks of each policy.
    2. allowed fetches of action distribution of the target network of each
    policy.

    Note that this Mixin is with policy. That is to say, the target network
    of each policy is maintain by their own. After each training iteration, all
    policy will update their own target network.
    """

    def __init__(self, obs_space, action_space, config):
        assert config[DELAY_UPDATE]

        # Build the target network of this policy.
        _, logit_dim = ModelCatalog.get_action_dist(
            action_space, config["model"]
        )
        self.target_model = ModelCatalog.get_model_v2(
            obs_space,
            action_space,
            logit_dim,
            config["model"],
            name="target_func",
            framework="tf"
        )
        self.model_vars = self.model.variables()
        self.target_model_vars = self.target_model.variables()

        self.get_session().run(
            tf.variables_initializer(self.target_model_vars))

        # Here is the delayed update mechanism.
        self.tau_value = config.get("tau")
        self.tau = tf.placeholder(tf.float32, (), name="tau")
        assign_ops = []
        assert len(self.model_vars) == len(self.target_model_vars)
        for var, var_target in zip(self.model_vars, self.target_model_vars):
            assign_ops.append(
                var_target.
                    assign(self.tau * var + (1.0 - self.tau) * var_target)
            )
        self.update_target_expr = tf.group(*assign_ops)

        @make_tf_callable(self.get_session(), True)
        def compute_clone_network_logits(ob):
            feed_dict = {
                SampleBatch.CUR_OBS: tf.convert_to_tensor(ob),
                "is_training": tf.convert_to_tensor(False)
            }
            model_out, _ = self.target_model(feed_dict)
            return model_out

        self._compute_clone_network_logits = compute_clone_network_logits

    def update_target_network(self, tau=None):
        """Delayed update the target network."""
        tau = tau or self.tau_value
        return self.get_session().run(
            self.update_target_expr, feed_dict={self.tau: tau}
        )

    def update_target(self, tau=None):
        import warnings
        warnings.warn(
            "Please use update_target_network! Current update_target function "
            "is deprecated.",
            DeprecationWarning)
        return self.update_target_network(tau)


def setup_mixins_dice(policy, action_space, obs_space, config):
    original_setup_mixins(policy, action_space, obs_space, config)
    DiversityValueNetworkMixin.__init__(policy, obs_space, action_space,
                                        config)
    ComputeDiversityMixin.__init__(policy)


def setup_late_mixins(policy, obs_space, action_space, config):
    if config[DELAY_UPDATE]:
        TargetNetworkMixin.__init__(policy, obs_space, action_space, config)


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
    after_init=setup_late_mixins,
    mixins=[
        LearningRateSchedule,
        # EntropyCoeffSchedule,
        KLCoeffMixin,
        TargetNetworkMixin,
        ValueNetworkMixin,
        DiversityValueNetworkMixin,
        ComputeDiversityMixin,

    ]
)
