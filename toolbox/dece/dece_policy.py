import logging

from ray.rllib.agents.impala.vtrace_policy import BEHAVIOUR_LOGITS
from ray.rllib.agents.ppo.ppo_policy import setup_mixins, \
    EntropyCoeffSchedule, \
    BEHAVIOUR_LOGITS, kl_and_loss_stats, PPOTFPolicy, KLCoeffMixin, \
    ValueNetworkMixin
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import LearningRateSchedule
from ray.rllib.utils.explained_variance import explained_variance
from ray.rllib.utils.tf_ops import make_tf_callable

from toolbox.dece.dece_loss import loss_dece, tnb_gradients
from toolbox.dece.dece_postprocess import postprocess_dece, MY_LOGIT
from toolbox.dece.utils import *
from toolbox.distance import get_kl_divergence

logger = logging.getLogger(__name__)

POLICY_SCOPE = "func"
TARGET_POLICY_SCOPE = "target_func"


def wrap_stats_ceppo(policy, train_batch):
    ret = kl_and_loss_stats(policy, train_batch)
    if hasattr(policy.loss_obj, "stats"):
        assert isinstance(policy.loss_obj.stats, dict)
        ret.update(policy.loss_obj.stats)
    return ret


def grad_stats_fn(policy, batch, grads):
    if policy.config[USE_BISECTOR] and (not policy.config[I_AM_CLONE]):
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
    ret = {BEHAVIOUR_LOGITS: policy.model.last_output()}
    if not policy.config[USE_VTRACE]:
        ret[SampleBatch.VF_PREDS] = policy.model.value_function()
    if policy.config[USE_DIVERSITY_VALUE_NETWORK]:
        ret[NOVELTY_VALUES] = policy.model.novelty_value_function()
    return ret


def kl_and_loss_stats_modified(policy, train_batch):
    if policy.config[I_AM_CLONE]:
        return {}
    ret = {
        "cur_kl_coeff": tf.cast(policy.kl_coeff, tf.float64),
        "cur_lr": tf.cast(policy.cur_lr, tf.float64),
        "total_loss": policy.loss_obj.loss,
        "policy_loss": policy.loss_obj.mean_policy_loss,
        "vf_loss": policy.loss_obj.mean_vf_loss,
        "kl": policy.loss_obj.mean_kl,
        "entropy": policy.loss_obj.mean_entropy,
        "entropy_coeff": tf.cast(policy.entropy_coeff, tf.float64),
    }
    if not policy.config[USE_VTRACE]:
        ret["vf_explained_var"] = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS],
            policy.model.value_function()
        ),

    if not policy.config[DIVERSITY_ENCOURAGING]:
        return ret
    ret.update(
        {
            "novelty_total_loss": policy.novelty_loss_obj.loss,
            "novelty_policy_loss": policy.novelty_loss_obj.mean_policy_loss,
            "novelty_vf_loss": policy.novelty_loss_obj.mean_vf_loss,
            "novelty_kl": policy.novelty_loss_obj.mean_kl,
            "novelty_entropy": policy.novelty_loss_obj.mean_entropy,
            "novelty_reward_mean": policy.novelty_reward_mean,
            "debug_ratio": policy.debug_ratio,
            "abs_advantage": policy.abs_advantage
        }
    )
    if policy.config[USE_DIVERSITY_VALUE_NETWORK
                     ] and not policy.config[USE_VTRACE]:
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
        assert not self.config[I_AM_CLONE]
        assert self.config[DELAY_UPDATE]
        self.policies_pool = {
            agent_name: other_policy
            for agent_name, other_policy in policies_pool.items()
            # if agent_name != my_name
        }  # Since it must in DELAY_UPDATE mode, we allow reuse all polices.
        self.num_of_policies = len(self.policies_pool)
        self.initialized_policies_pool = True

    def compute_novelty(self, my_batch, others_batches, use_my_logits):
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
            for other_name, other_policy in self.policies_pool.items():
                logits = other_policy._compute_clone_network_logits(
                    my_batch[SampleBatch.CUR_OBS]
                )
                replays[other_name] = logits
        else:
            for other_name, (other_policy, _) in others_batches.items():
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


def get_batch_divisibility_req(policy):
    if policy.config[USE_VTRACE]:
        return policy.config['sample_batch_size']
    else:
        return 1


class TargetNetworkMixin(object):
    def __init__(self, obs_space, action_space, config):
        """Target Network is updated by the master learner every
        trainer.update_target_frequency steps. All worker batches
        are importance sampled w.r. to the target network to ensure
        a more stable pi_old in PPO.
        """
        assert config[DELAY_UPDATE]
        _, logit_dim = ModelCatalog.get_action_dist(
            action_space, config["model"]
        )
        self.target_model = ModelCatalog.get_model_v2(
            obs_space,
            action_space,
            logit_dim,
            config["model"],
            name=TARGET_POLICY_SCOPE,
            framework="tf"
        )

        self.model_vars = self.model.variables()
        self.target_model_vars = self.target_model.variables()

        self.get_session().run(tf.initialize_variables(self.target_model_vars))

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
            # def compute_clone_network_logits(ob, prev_action, prev_reward):
            # We do not support recurrent network now.
            feed_dict = {
                SampleBatch.CUR_OBS: tf.convert_to_tensor(ob),
                # SampleBatch.PREV_REWARDS: tf.convert_to_tensor(
                #     prev_reward),
                "is_training": tf.convert_to_tensor(False)
            }
            # if prev_action is not None:
            #     feed_dict[SampleBatch.PREV_ACTIONS] = tf.convert_to_tensor(
            #         prev_action)
            model_out, _ = self.target_model(feed_dict)
            return model_out

        self._compute_clone_network_logits = compute_clone_network_logits

    def update_clone_network(self, tau=None):
        tau = tau or self.tau_value
        return self.get_session().run(
            self.update_target_expr, feed_dict={self.tau: tau}
        )


def setup_late_mixins(policy, obs_space, action_space, config):
    if config[DELAY_UPDATE]:
        TargetNetworkMixin.__init__(policy, obs_space, action_space, config)


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
    after_init=setup_late_mixins,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin, NoveltyValueNetworkMixin, ComputeNoveltyMixin,
        TargetNetworkMixin
    ],
    get_batch_divisibility_req=get_batch_divisibility_req,
)
