import copy
import json
import logging
import os
import pickle
from collections import OrderedDict

from ray.rllib.agents.ppo.ppo import PPOTFPolicy, DEFAULT_CONFIG
from ray.rllib.agents.ppo.ppo_tf_policy import setup_mixins, \
    ValueNetworkMixin, KLCoeffMixin, LearningRateSchedule, \
    EntropyCoeffSchedule, SampleBatch, \
    BEHAVIOUR_LOGITS, make_tf_callable, kl_and_loss_stats
from ray.rllib.evaluation.postprocessing import Postprocessing, discount
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.explained_variance import explained_variance

from toolbox.marl.utils import on_train_result
from toolbox.task_novelty_bisector.tnb_loss import tnb_gradients, tnb_loss
from toolbox.task_novelty_bisector.tnb_model import ActorDoubleCriticNetwork
from toolbox.task_novelty_bisector.tnb_utils import *
from toolbox.utils import merge_dicts

tf = try_import_tf()
logger = logging.getLogger(__name__)

tnb_default_config = merge_dicts(
    DEFAULT_CONFIG,
    dict(
        novelty_threshold=0.5,
        use_preoccupied_agent=False,
        disable_tnb=False,
        use_tnb_plus=True,
        checkpoint_dict="{}",  # use json to parse a dict into string.
        # disabling novelty value network can save the cost of extra NN and
        # prevents misleading novelty policy gradient.
        use_novelty_value_network=True,

        # Do not modified these parameters.
        distance_mode="min",
        tnb_plus_threshold=0.0,
        clip_novelty_gradient=False,
        use_second_component=True,
        model={"custom_model": "ActorDoubleCriticNetwork"},
        callbacks={"on_train_result": on_train_result}
    )
)

ModelCatalog.register_custom_model(
    "ActorDoubleCriticNetwork", ActorDoubleCriticNetwork
)


def get_action_mean(logits):
    return np.split(logits, 2, axis=1)[0]


def postprocess_tnb(policy, sample_batch, other_batches, episode):
    completed = sample_batch["dones"][-1]
    sample_batch[NOVELTY_REWARDS] = policy.compute_novelty(
        sample_batch, other_batches, episode
    )

    if completed:
        last_r_novelty = last_r = 0.0
    else:
        next_state = []
        for i in range(policy.num_state_tensors()):
            next_state.append([sample_batch["state_out_{}".format(i)][-1]])
        last_r = policy._value(
            sample_batch[SampleBatch.NEXT_OBS][-1],
            sample_batch[SampleBatch.ACTIONS][-1],
            sample_batch[SampleBatch.REWARDS][-1], *next_state
        )
        last_r_novelty = policy._novelty_value(
            sample_batch[SampleBatch.NEXT_OBS][-1],
            sample_batch[SampleBatch.ACTIONS][-1],
            sample_batch[NOVELTY_REWARDS][-1], *next_state
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
        rewards=sample_batch[NOVELTY_REWARDS],
        last_r=last_r_novelty,
        gamma=policy.config["gamma"],
        lambda_=policy.config["lambda"],
        values=sample_batch[NOVELTY_VALUES]
        if policy.config['use_novelty_value_network'] else None,
        use_gae=policy.config['use_novelty_value_network']
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
        value_target = np.zeros_like(advantage)
    advantage = advantage.copy().astype(np.float32)
    return advantage, value_target


def _restore_state(ckpt):
    wkload = pickle.load(open(ckpt, 'rb'))['worker']
    state = pickle.loads(wkload)['state']['default_policy']
    return state


class AgentPoolMixin(object):
    def __init__(self, checkpoint_dict, threshold, distance_mode='min'):
        self.checkpoint_dict = checkpoint_dict
        self.enable_novelty = (len(self.checkpoint_dict) != 0) and \
                              (not self.config['disable_tnb'])
        self.threshold = threshold
        assert distance_mode in ['min', 'max']
        self.distance_mode = distance_mode
        self.initialized_policies_pool = False
        self.policies_pool = OrderedDict()

    def _lazy_initialize(self):
        # remove checkpoint_dict, otherwise will create nested policies.
        tmp_config = copy.deepcopy(self.config)
        tmp_config["checkpoint_dict"] = "{}"
        # disable the private worker of each policy, to save resource.
        tmp_config.update(
            {
                "num_workers": 0,
                "num_cpus_per_worker": 0,
                "num_cpus_for_driver": 0.2,
                "num_gpus": 0.1,
            }
        )
        for i, (agent_name, checkpoint_info) in \
                enumerate(self.checkpoint_dict.items()):
            # build the policy and restore the weights.
            with tf.variable_scope(agent_name, reuse=tf.AUTO_REUSE):
                policy = TNBPolicy(
                    self.observation_space, self.action_space, tmp_config
                )
                if checkpoint_info is not None:
                    path = os.path.abspath(
                        os.path.expanduser(checkpoint_info['path'])
                    )
                    state = _restore_state(path)

                    old_agent_name = next(iter(state.keys())).split("/")[0]
                    policy.set_weights({
                        w_name.replace(old_agent_name, agent_name): w
                        for w_name, w in state.items()
                    })
                else:  # for test purpose
                    checkpoint_info = {'path': "N/A", 'reward': float('nan')}

            policy_info = {
                "policy": policy,
                "agent_name": agent_name,
                "checkpoint_path": checkpoint_info['path'],
                "reward": checkpoint_info['reward']
            }
            self.policies_pool[agent_name] = policy_info

        if self.config['use_preoccupied_agent'] and self.checkpoint_dict:
            best_agent = max(
                self.checkpoint_dict,
                key=lambda k: self.checkpoint_dict[k]['reward']
            )
            assert next(iter(self.checkpoint_dict.keys())) == best_agent
            self.set_weights(
                self.policies_pool[best_agent]['policy'].get_weights()
            )
            msg = (
                "We successfully restore current agent with "
                " best agent <{}>, it's reward {}. ".format(
                    best_agent, self.checkpoint_dict[best_agent]['reward']
                )
            )
            logger.info(msg)
            print(msg)

        self.num_of_policies = len(self.policies_pool)
        self.novelty_stat = RunningMean(self.num_of_policies)

        self.initialized_policies_pool = True

    def compute_novelty(self, sample_batch, other_batches=None, episode=None):
        state = sample_batch[SampleBatch.CUR_OBS]
        action = sample_batch[SampleBatch.ACTIONS]
        if not self.initialized_policies_pool:
            if not hasattr(self, "_loss_inputs"):
                return np.zeros((action.shape[0],), dtype=np.float32)

        if not self.enable_novelty:
            return np.zeros((action.shape[0],), dtype=np.float32)

        assert self.initialized_policies_pool, self.policies_pool

        diff_list = []
        for i, (key, policy_dict) in enumerate(self.policies_pool.items()):
            policy = policy_dict['policy']
            _, _, info = policy.compute_actions(state)
            other_action = get_action_mean(info[BEHAVIOUR_LOGITS])
            diff_list.append(np.linalg.norm(other_action - action, axis=1))

        per_policy_novelty = self.novelty_stat(diff_list)
        if self.distance_mode == 'min':
            min_novel = np.min(per_policy_novelty, axis=0)
            # self.novelty_recorder / self.novelty_recorder_len)
            return min_novel - self.threshold
        elif self.distance_mode == 'max':
            max_novel = np.max(per_policy_novelty, axis=0)
            # self.novelty_recorder / self.novelty_recorder_len)
            return max_novel - self.threshold
        else:
            raise NotImplementedError()


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


def setup_mixins_tnb(policy, action_space, obs_space, config):
    setup_mixins(policy, action_space, obs_space, config)
    NoveltyValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    assert isinstance(config["checkpoint_dict"], str)
    checkpoint_dict = json.loads(config["checkpoint_dict"])
    AgentPoolMixin.__init__(
        policy, checkpoint_dict, config['novelty_threshold'],
        config['distance_mode']
    )


def additional_fetches(policy):
    """Adds value function and logits outputs to experience train_batches."""
    ret = {
        SampleBatch.VF_PREDS: policy.model.value_function(),
        BEHAVIOUR_LOGITS: policy.model.last_output()
    }
    if policy.config['use_novelty_value_network']:
        ret[NOVELTY_VALUES] = policy.model.novelty_value_function()
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


TNBPolicy = PPOTFPolicy.with_updates(
    name="TNBPolicy",
    get_default_config=lambda: tnb_default_config,
    before_loss_init=setup_mixins_tnb,
    extra_action_fetches_fn=additional_fetches,
    postprocess_fn=postprocess_tnb,
    loss_fn=tnb_loss,
    stats_fn=kl_and_loss_stats_modified,
    gradients_fn=tnb_gradients,
    grad_stats_fn=grad_stats_fn,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin, NoveltyValueNetworkMixin, AgentPoolMixin
    ]
)
