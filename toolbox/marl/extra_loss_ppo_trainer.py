import logging

import numpy as np
import tensorflow as tf
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG, validate_config, \
    PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy, \
    LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin, \
    ValueNetworkMixin, ppo_surrogate_loss, postprocess_ppo_gae, setup_mixins
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.models.tf.tf_action_dist import DiagGaussian
from ray.rllib.optimizers import SyncSamplesOptimizer
from ray.rllib.policy.rnn_sequencing import chop_into_sequences
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch
from ray.rllib.utils.explained_variance import explained_variance

from toolbox.modified_rllib.multi_gpu_optimizer import \
    LocalMultiGPUOptimizerModified
from toolbox.utils import merge_dicts

logger = logging.getLogger(__name__)

# Frozen logits of the policy that computed the action
BEHAVIOUR_LOGITS = "behaviour_logits"
OPPONENT_OBS = "opponent_obs"
OPPONENT_ACTION = "opponent_action"

PEER_ACTION = "other_replay"
JOINT_OBS = "joint_obs"
# NO_SPLIT_OBS = "no_split_obs"

mixin_list = [
    LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin, ValueNetworkMixin
]

extra_loss_ppo_default_config = merge_dicts(
    DEFAULT_CONFIG,
    dict(
        novelty_loss_param=0.5,
        joint_dataset_sample_batch_size=200,
        novelty_mode="mean",
        use_joint_dataset=True
    )
)


def postprocess_ppo_gae_modified(
        policy, sample_batch, other_agent_batches=None, episode=None
):
    """This function add extra placeholder, by creating new entries in batch
    which the following RLLib procedure would detect and create placeholder
    based on the shape of them."""
    batch = postprocess_ppo_gae(
        policy, sample_batch, other_agent_batches, episode
    )
    if not policy.loss_initialized():
        batch[JOINT_OBS] = np.zeros_like(
            sample_batch[SampleBatch.CUR_OBS], dtype=np.float32
        )
        batch[PEER_ACTION] = np.zeros_like(
            sample_batch[SampleBatch.ACTIONS], dtype=np.float32
        )  # peer_action is needed no matter use joint_dataset or not.
    return batch


class AddLossMixin(object):
    """When training, this Mixin add the sharing data into feed_dict.

    Originally, it's a function of tf_policy, so we copied some codes from
    tf_policy.py"""

    def __init__(self, config):
        if "novelty_loss_param_init" in config:
            self.novelty_loss_param = config["novelty_loss_param_init"]
        elif "novelty_loss_param" in config:
            self.novelty_loss_param = config["novelty_loss_param"]
        else:
            logger.warning(
                "You Do Not Specify 'novelty_loss_param_init' or"
                " 'novelty_loss_param' in config, so we do not"
                " define policy.novelty_loss_param"
            )
        self._AddLossMixin_initialized = True

    def _get_loss_inputs_dict(
            self, batch, shuffle, cross_policy_obj, policy_id=None
    ):
        """When training, add the required data into the feed_dict."""
        feed_dict = {}

        if hasattr(self, "_AddLossMixin_initialized"):
            assert self._AddLossMixin_initialized
            # parse the cross-policy info and put in feed_dict.
            joint_obs_ph = self._loss_input_dict[JOINT_OBS]
            feed_dict[joint_obs_ph] = cross_policy_obj[JOINT_OBS]

            replay_ph = self._loss_input_dict[PEER_ACTION]
            feed_dict[replay_ph] = np.concatenate(
                [
                    act for pid, act in cross_policy_obj[PEER_ACTION].items()
                    if pid != policy_id
                ]
            )  # exclude policy itself action
        """The below codes are copied from rllib. """
        if self._batch_divisibility_req > 1:
            meets_divisibility_reqs = (
                    len(batch[SampleBatch.CUR_OBS]) %
                    self._batch_divisibility_req == 0
                    and max(batch[SampleBatch.AGENT_INDEX]) == 0
            )  # not multiagent
        else:
            meets_divisibility_reqs = True
        # Simple case: not RNN nor do we need to pad
        if not self._state_inputs and meets_divisibility_reqs:
            if shuffle:
                batch.shuffle()
            for k, ph in self._loss_inputs:
                if k in batch:  # Attention! We add a condition here.
                    feed_dict[ph] = batch[k]
            return feed_dict
        if self._state_inputs:
            max_seq_len = self._max_seq_len
            dynamic_max = True
        else:
            max_seq_len = self._batch_divisibility_req
            dynamic_max = False
        # RNN or multi-agent case
        feature_keys = [k for k, v in self._loss_inputs]
        state_keys = [
            "state_in_{}".format(i) for i in range(len(self._state_inputs))
        ]
        feature_sequences, initial_states, seq_lens = chop_into_sequences(
            batch[SampleBatch.EPS_ID],
            batch[SampleBatch.UNROLL_ID],
            batch[SampleBatch.AGENT_INDEX], [batch[k] for k in feature_keys],
            [batch[k] for k in state_keys],
            max_seq_len,
            dynamic_max=dynamic_max,
            shuffle=shuffle
        )
        for k, v in zip(feature_keys, feature_sequences):
            feed_dict[self._loss_input_dict[k]] = v
        for k, v in zip(state_keys, initial_states):
            feed_dict[self._loss_input_dict[k]] = v
        feed_dict[self._seq_lens] = seq_lens
        return feed_dict


def novelty_loss_mse(policy, model, dist_class, train_batch):
    mode = policy.config['novelty_mode']
    obs_ph = train_batch[JOINT_OBS]
    if dist_class == DiagGaussian:
        discrete = False
    # elif dist_class == Categorical:
    #     discrete = True
    else:
        raise NotImplementedError(
            "Only support DiagGaussian, --Categorical(WIP)-- distribution."
        )

    # The ret_act is the 'behaviour_logits'
    ret_act, _ = model.base_model(obs_ph)
    if discrete:
        raise NotImplementedError("We do not implement discrete action space.")
    else:
        my_act = tf.split(ret_act, 2, axis=1)[0]
        peer_act_ph = train_batch[PEER_ACTION]
        flatten = tf.reshape(my_act, [-1])
        other_act = tf.reshape(peer_act_ph, [-1, tf.shape(flatten)[0]])
        subtract = tf.subtract(flatten, other_act)
        normalized = tf.norm(subtract, axis=1)  # normalized for each policies.
        if mode == "mean":
            nov_loss = -tf.reduce_mean(normalized)
        elif mode == "min":
            nov_loss = -tf.reduce_min(normalized)
        else:
            nov_loss = -tf.reduce_max(normalized)
    policy.novelty_loss = nov_loss
    return nov_loss


def extra_loss_ppo_loss(policy, model, dist_class, train_batch):
    """Add novelty loss with original ppo loss"""
    original_loss = ppo_surrogate_loss(policy, model, dist_class, train_batch)
    nov_loss = novelty_loss_mse(policy, model, dist_class, train_batch)
    alpha = policy.novelty_loss_param
    total_loss = (1 - alpha) * original_loss + alpha * nov_loss
    policy.total_loss = total_loss
    return total_loss


def kl_and_loss_stats_modified(policy, train_batch):
    ret = kl_and_loss_stats_without_total_loss(policy, train_batch)
    ret.update(total_loss=policy.total_loss)
    return ret


def kl_and_loss_stats_without_total_loss(policy, train_batch):
    ret = {
        "cur_kl_coeff": tf.cast(policy.kl_coeff, tf.float64),
        "cur_lr": tf.cast(policy.cur_lr, tf.float64),
        "policy_loss": policy.loss_obj.mean_policy_loss,
        "vf_loss": policy.loss_obj.mean_vf_loss,
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS],
            policy.model.value_function()
        ),
        "kl": policy.loss_obj.mean_kl,
        "entropy": policy.loss_obj.mean_entropy,
        "entropy_coeff": tf.cast(policy.entropy_coeff, tf.float64),
    }
    if hasattr(policy, "novelty_loss"):
        ret["novelty_loss"] = policy.novelty_loss
    return ret


def get_cross_policy_object(multi_agent_batch, self_optimizer):
    """Add contents into cross_policy_object, which passed to each policy."""
    config = self_optimizer.workers._remote_config

    if not config["use_joint_dataset"]:
        joint_obs = SampleBatch.concat_samples(
            list(multi_agent_batch.policy_batches.values())
        )[SampleBatch.CUR_OBS]
    else:
        sample_size = config.get("joint_dataset_sample_batch_size")
        assert sample_size is not None, "You should specify the value of: " \
                                        "joint_dataset_sample_batch_size " \
                                        "in config!"
        samples = [multi_agent_batch]
        count_dict = {
            k: v.count
            for k, v in multi_agent_batch.policy_batches.items()
        }
        for k in self_optimizer.workers.local_worker().policy_map.keys():
            if k not in count_dict:
                count_dict[k] = 0

        while any([v < sample_size for v in count_dict.values()]):
            tmp_batch = self_optimizer.workers.local_worker().sample()
            samples.append(tmp_batch)
            for k, v in tmp_batch.policy_batches.items():
                assert k in count_dict, count_dict
                count_dict[k] += v.count
        multi_agent_batch = MultiAgentBatch.concat_samples(samples)

        joint_obs = []
        pid_list = []
        for pid, batch in multi_agent_batch.policy_batches.items():
            batch.shuffle()
            assert batch.count >= sample_size, batch
            joint_obs.append(batch.slice(0, sample_size)['obs'])
            pid_list.append(pid)
        joint_obs = np.concatenate(joint_obs)

    def _replay(policy, pid):
        act, _, infos = policy.compute_actions(joint_obs)
        return pid, act, infos

    # ATTENTION!!! Here is MYSELF replaying JOINT OBSERVATION
    ret = {
        pid: act
        for pid, act, infos in
        self_optimizer.workers.local_worker().foreach_policy(_replay)
    }
    return {JOINT_OBS: joint_obs, PEER_ACTION: ret}


def validate_config_basic(config):
    assert "joint_dataset_sample_batch_size" in config
    assert "use_joint_dataset" in config
    assert "novelty_mode" in config
    assert config["novelty_mode"] in ["mean", "min", "max"]
    validate_config(config)


def validate_config_modified(config):
    assert "novelty_loss_param" in config
    validate_config_basic(config)


def choose_policy_optimizer(workers, config):
    if config["simple_optimizer"]:
        return SyncSamplesOptimizer(
            workers,
            num_sgd_iter=config["num_sgd_iter"],
            train_batch_size=config["train_batch_size"],
            sgd_minibatch_size=config["sgd_minibatch_size"],
            standardize_fields=["advantages"]
        )

    no_split_list = [JOINT_OBS, PEER_ACTION]

    return LocalMultiGPUOptimizerModified(
        workers,
        no_split_list=no_split_list,
        process_multiagent_batch_fn=get_cross_policy_object,
        sgd_batch_size=config["sgd_minibatch_size"],
        num_sgd_iter=config["num_sgd_iter"],
        num_gpus=config["num_gpus"],
        sample_batch_size=config["sample_batch_size"],
        num_envs_per_worker=config["num_envs_per_worker"],
        train_batch_size=config["train_batch_size"],
        standardize_fields=["advantages"],
        shuffle_sequences=config["shuffle_sequences"]
    )


def setup_mixins_modified(policy, obs_space, action_space, config):
    AddLossMixin.__init__(policy, config)
    setup_mixins(policy, obs_space, action_space, config)


ExtraLossPPOTFPolicy = PPOTFPolicy.with_updates(
    name="ExtraLossPPOTFPolicy",
    get_default_config=lambda: extra_loss_ppo_default_config,
    postprocess_fn=postprocess_ppo_gae_modified,
    stats_fn=kl_and_loss_stats_modified,
    loss_fn=extra_loss_ppo_loss,
    before_loss_init=setup_mixins_modified,
    mixins=mixin_list + [AddLossMixin]
)

ExtraLossPPOTrainer = PPOTrainer.with_updates(
    name="ExtraLossPPO",
    default_config=extra_loss_ppo_default_config,
    validate_config=validate_config_modified,
    default_policy=ExtraLossPPOTFPolicy,
    make_policy_optimizer=choose_policy_optimizer
)

if __name__ == '__main__':
    from toolbox.marl.test_extra_loss import test_extra_loss_ppo_trainer1

    test_extra_loss_ppo_trainer1(True)
