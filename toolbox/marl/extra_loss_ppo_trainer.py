import logging

import numpy as np
import tensorflow as tf
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG, validate_config
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy, \
    LearningRateSchedule, \
    EntropyCoeffSchedule, KLCoeffMixin, \
    ValueNetworkMixin, ppo_surrogate_loss
from ray.rllib.evaluation.postprocessing import compute_advantages, \
    Postprocessing
from ray.rllib.optimizers import SyncSamplesOptimizer
from ray.rllib.policy.rnn_sequencing import chop_into_sequences
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch
from ray.rllib.utils.explained_variance import explained_variance
from ray.tune.util import merge_dicts

from toolbox import initialize_ray
from toolbox.marl import MultiAgentEnvWrapper, on_train_result
from toolbox.modified_rllib.multi_gpu_optimizer import \
    LocalMultiGPUOptimizerModified
from toolbox.utils import get_local_dir

logger = logging.getLogger(__name__)

# Frozen logits of the policy that computed the action
BEHAVIOUR_LOGITS = "behaviour_logits"
OPPONENT_OBS = "opponent_obs"
OPPONENT_ACTION = "opponent_action"

PEER_ACTION = "other_replay"
JOINT_OBS = "joint_dataset"
NO_SPLIT_ACTION = "no_split_actions"

extra_loss_ppo_default_config = merge_dicts(
    DEFAULT_CONFIG,
    dict(
        novelty_loss_param=0.5,
        joint_dataset_sample_batch_size=200,
        novelty_mode="mean",
        use_joint_dataset=True
    )
)


def postprocess_ppo_gae(
        policy, sample_batch, other_agent_batches=None, episode=None
):
    completed = sample_batch["dones"][-1]
    if completed:
        last_r = 0.0
    else:
        next_state = []
        for i in range(policy.num_state_tensors()):
            next_state.append([sample_batch["state_out_{}".format(i)][-1]])
        last_r = policy._value(
            sample_batch[SampleBatch.NEXT_OBS][-1],
            sample_batch[SampleBatch.ACTIONS][-1],
            sample_batch[SampleBatch.REWARDS][-1], *next_state
        )
    batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"]
    )

    if not policy.loss_initialized():
        if policy.config["use_joint_dataset"]:
            batch[JOINT_OBS] = np.zeros_like(
                sample_batch[SampleBatch.CUR_OBS], dtype=np.float32
            )
        else:
            batch[NO_SPLIT_ACTION] = np.zeros_like(
                sample_batch[SampleBatch.ACTIONS], dtype=np.float32
            )
        batch[PEER_ACTION] = np.zeros_like(
            sample_batch[SampleBatch.ACTIONS], dtype=np.float32
        )  # peer_action is needed no matter use joint_dataset or not.
    return batch


class AddLossMixin(object):
    """Copied from tf_policy.py"""

    def _get_loss_inputs_dict(
            self, batch, shuffle, cross_policy_obj, policy_id=None
    ):
        """Return a feed dict from a batch.

        Arguments:
            batch (SampleBatch): batch of data to derive inputs from
            shuffle (bool): whether to shuffle batch sequences. Shuffle may
                be done in-place. This only makes sense if you're further
                applying minibatch SGD after getting the outputs.

        Returns:
            feed dict of data
        """
        feed_dict = {}

        # What we modified
        if self.config["use_joint_dataset"]:
            # parse the cross-policy info and put in feed_dict.
            replay_ph = self._loss_input_dict[PEER_ACTION]
            joint_obs_ph = self._loss_input_dict[JOINT_OBS]
            feed_dict[joint_obs_ph] = cross_policy_obj[JOINT_OBS]
            concat_replay_act = np.concatenate(
                [
                    act for pid, act in cross_policy_obj[PEER_ACTION].items()
                    if pid != policy_id
                ]
            )  # exclude policy itself action
            feed_dict[replay_ph] = concat_replay_act
        else:
            replay_ph = self._loss_input_dict[PEER_ACTION]
            concat_replay_act = np.concatenate(
                [
                    act for pid, act in cross_policy_obj[policy_id].items()
                    if pid != policy_id
                ]
            )  # exclude policy itself action
            feed_dict[replay_ph] = concat_replay_act
            feed_dict[self._loss_input_dict[NO_SPLIT_ACTION]] = \
                batch['actions']

        # The below codes are copied from rllib.
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
                if k in batch:
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


def norm(my_act, other_act, mode="mean"):
    # the other_act should exclude itself.
    subtract = tf.subtract(my_act, other_act)
    normalized = tf.norm(subtract, axis=1)  # normalized for each policies.
    if mode == "mean":
        return tf.reduce_mean(normalized)
    elif mode == "min":
        return tf.reduce_min(normalized)
    else:
        return tf.reduce_max(normalized)


def novelty_loss(policy, model, dist_class, train_batch):
    mode = policy.config['novelty_mode']
    if policy.config.get("use_joint_dataset"):
        joint_obs_ph = train_batch[JOINT_OBS]
        ret_act, _ = model.base_model(joint_obs_ph)
        my_act = tf.split(ret_act, 2, axis=1)[0]
    else:
        my_act = train_batch[NO_SPLIT_ACTION]
    peer_act_ph = train_batch[PEER_ACTION]
    flatten = tf.reshape(my_act, [-1])
    other_act = tf.reshape(peer_act_ph, [-1, tf.shape(flatten)[0]])
    nov_loss = -norm(flatten, other_act, mode)
    policy.novelty_loss = nov_loss
    return nov_loss


def extra_loss_ppo_loss(policy, model, dist_class, train_batch):
    """Add novelty loss with original ppo loss"""
    original_loss = ppo_surrogate_loss(policy, model, dist_class, train_batch)
    nov_loss = novelty_loss(policy, model, dist_class, train_batch)
    alpha = policy.config["novelty_loss_param"]
    total_loss = (1 - alpha) * original_loss + alpha * nov_loss
    policy.total_loss = total_loss
    return total_loss


def kl_and_loss_stats_modified(policy, train_batch):
    ret = kl_and_loss_stats_without_total_loss(policy, train_batch)
    ret.update(total_loss=policy.total_loss)
    return ret


def kl_and_loss_stats_without_total_loss(policy, train_batch):
    return {
        "novelty_loss": policy.novelty_loss,
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


def cross_policy_object_use_joint_dataset(multi_agent_batch, self_optimizer):
    """Add contents into cross_policy_object, which passed to each policy."""
    config = self_optimizer.workers._remote_config
    assert config["use_joint_dataset"], "You should set use_joint_dataset " \
                                        "to True"
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

    ret = {
        pid: act
        for pid, act, infos in
        self_optimizer.workers.local_worker().foreach_policy(_replay)
    }
    return {JOINT_OBS: joint_obs, PEER_ACTION: ret}


def cross_policy_object_without_joint_dataset(
        multi_agent_batch, self_optimizer
):
    """Add contents into cross_policy_object, which passed to each policy."""
    config = self_optimizer.workers._remote_config
    assert not config["use_joint_dataset"]
    return_dict = {}
    # replay All possible observations for All agents
    for pid, batch in multi_agent_batch.policy_batches.items():
        ret = {}

        def _replay(policy, replay_pid):
            if replay_pid == pid:
                return None, None
            act, _, infos = policy.compute_actions(batch['obs'])
            return replay_pid, act

        for replay_pid, act in \
                self_optimizer.workers.local_worker().foreach_policy(_replay):
            if act is None:
                continue
            ret[replay_pid] = act
        return_dict[pid] = ret
    return return_dict


def validate_config_basic(config):
    assert "joint_dataset_sample_batch_size" in config
    assert "use_joint_dataset" in config
    assert "novelty_mode" in config
    assert config["novelty_mode"] in ["mean", "min", "max"]
    validate_config(config)


def validate_config_modified(config):
    assert "novelty_param" in config
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

    split_list = [JOINT_OBS, PEER_ACTION] \
        if config['use_joint_dataset'] else [PEER_ACTION, NO_SPLIT_ACTION]

    return LocalMultiGPUOptimizerModified(
        workers,
        split_list,
        cross_policy_object_use_joint_dataset if config["use_joint_dataset"]
        else cross_policy_object_without_joint_dataset,
        sgd_batch_size=config["sgd_minibatch_size"],
        num_sgd_iter=config["num_sgd_iter"],
        num_gpus=config["num_gpus"],
        sample_batch_size=config["sample_batch_size"],
        num_envs_per_worker=config["num_envs_per_worker"],
        train_batch_size=config["train_batch_size"],
        standardize_fields=["advantages"],
        shuffle_sequences=config["shuffle_sequences"]
    )


ExtraLossPPOTFPolicy = PPOTFPolicy.with_updates(
    name="ExtraLossPPOTFPolicy",
    get_default_config=lambda: extra_loss_ppo_default_config,
    postprocess_fn=postprocess_ppo_gae,
    stats_fn=kl_and_loss_stats_modified,
    loss_fn=extra_loss_ppo_loss,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin, AddLossMixin
    ]
)

ExtraLossPPOTrainer = PPOTrainer.with_updates(
    name="ExtraLossPPO",
    default_config=extra_loss_ppo_default_config,
    validate_config=validate_config_modified,
    default_policy=ExtraLossPPOTFPolicy,
    make_policy_optimizer=choose_policy_optimizer
)


def test_extra_loss_ppo_trainer_use_joint_dataset(extra_config=None):
    num_agents = 5
    num_gpus = 0

    # This is only test code.
    initialize_ray(test_mode=True, local_mode=False, num_gpus=num_gpus)

    policy_names = ["ppo_agent{}".format(i) for i in range(num_agents)]

    env_config = {"env_name": "BipedalWalker-v2", "agent_ids": policy_names}
    env = MultiAgentEnvWrapper(env_config)
    config = {
        "env": MultiAgentEnvWrapper,
        "env_config": env_config,
        "num_gpus": num_gpus,
        "log_level": "DEBUG",
        "joint_dataset_sample_batch_size": 37,
        "multiagent": {
            "policies": {
                i: (None, env.observation_space, env.action_space, {})
                for i in policy_names
            },
            "policy_mapping_fn": lambda x: x,
        },
        "callbacks": {
            "on_train_result": on_train_result
        },
    }
    if extra_config:
        config.update(extra_config)

    tune.run(
        ExtraLossPPOTrainer,
        local_dir=get_local_dir(),
        name="DELETEME_TEST_extra_loss_ppo_trainer",
        stop={"timesteps_total": 5000},
        config=config
    )


def test_extra_loss_ppo_trainer_without_joint_dataset():
    test_extra_loss_ppo_trainer_use_joint_dataset({"use_joint_dataset": False})


if __name__ == '__main__':
    test_extra_loss_ppo_trainer_use_joint_dataset()
    test_extra_loss_ppo_trainer_without_joint_dataset()
