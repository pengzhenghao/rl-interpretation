import logging
from collections import defaultdict

import numpy as np
import tensorflow as tf
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy, PPOLoss, \
    LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin, ValueNetworkMixin
from ray.rllib.evaluation.postprocessing import compute_advantages, \
    Postprocessing
from ray.rllib.optimizers import SyncSamplesOptimizer
from ray.rllib.policy.rnn_sequencing import chop_into_sequences
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch
from ray.rllib.policy.tf_policy import ACTION_LOGP
from ray.rllib.utils.explained_variance import explained_variance

from toolbox.marl import MultiAgentEnvWrapper
from toolbox.modified_rllib.multi_gpu_optimizer import \
    LocalMultiGPUOptimizerModified

logger = logging.getLogger(__name__)

# Frozen logits of the policy that computed the action
BEHAVIOUR_LOGITS = "behaviour_logits"
OPPONENT_OBS = "opponent_obs"
OPPONENT_ACTION = "opponent_action"

PEER_ACTION = "other_replay"
JOINT_OBS = "joint_dataset"


def postprocess_ppo_gae(policy,
                        sample_batch,
                        other_agent_batches=None,
                        episode=None):
    completed = sample_batch["dones"][-1]
    if completed:
        last_r = 0.0
    else:
        next_state = []
        for i in range(policy.num_state_tensors()):
            next_state.append([sample_batch["state_out_{}".format(i)][-1]])
        last_r = policy._value(sample_batch[SampleBatch.NEXT_OBS][-1],
                               sample_batch[SampleBatch.ACTIONS][-1],
                               sample_batch[SampleBatch.REWARDS][-1],
                               *next_state)
    batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"])

    if not policy.loss_initialized():
        # add some variable in batch to initialize the placeholders.
        batch[JOINT_OBS] = np.zeros_like(
            sample_batch[SampleBatch.CUR_OBS], dtype=np.float32)
        batch[PEER_ACTION] = np.zeros_like(
            sample_batch[SampleBatch.ACTIONS], dtype=np.float32)

    return batch


class AddLossMixin(object):
    """Copied from tf_policy.py"""

    def _get_loss_inputs_dict(self, batch, shuffle, cross_policy_obj):
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

        # parse the cross-policy info and put in feed_dict <- What we modified
        joint_obs_ph = self._loss_input_dict[JOINT_OBS]
        feed_dict[joint_obs_ph] = cross_policy_obj[JOINT_OBS]

        replay_ph = self._loss_input_dict[PEER_ACTION]
        feed_dict[replay_ph] = np.concatenate(
            list(cross_policy_obj[PEER_ACTION].values())
        )

        assert feed_dict[replay_ph].shape[0] % feed_dict[joint_obs_ph].shape[
            0] == 0

        # The below codes are copied from rllib.

        if self._batch_divisibility_req > 1:
            meets_divisibility_reqs = (
                    len(batch[SampleBatch.CUR_OBS]) %
                    self._batch_divisibility_req == 0
                    and max(
                batch[SampleBatch.AGENT_INDEX]) == 0)  # not multiagent
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
            shuffle=shuffle)
        for k, v in zip(feature_keys, feature_sequences):
            feed_dict[self._loss_input_dict[k]] = v
        for k, v in zip(state_keys, initial_states):
            feed_dict[self._loss_input_dict[k]] = v
        feed_dict[self._seq_lens] = seq_lens
        return feed_dict


# def reshape(obs, act):
#     return tf.reshape(act, [-1, tf.shape(obs)[0], tf.shape(act)[1]])


def norm(my_act, other_act):
    flatten = tf.reshape(my_act, [-1])
    other_act = tf.reshape(other_act,
                           [-1, tf.shape(my_act)[0] * tf.shape(my_act)[1]])
    print("IN NORM", flatten.shape, other_act.shape)
    return tf.reduce_mean(tf.subtract(flatten, other_act) ** 2)


def ppo_surrogate_loss(policy, model, dist_class, train_batch):
    """Copied from PPOTFPolicy"""
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
        train_batch[ACTION_LOGP],
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

    loss = policy.loss_obj.loss

    # The below codes is what we append
    joint_obs_ph = train_batch[JOINT_OBS]
    peer_act_ph = train_batch[PEER_ACTION]

    logger.debug(
        "The joint_obs_ph shape: {}, the peer_act_ph shape: {}".format(
            joint_obs_ph.shape, peer_act_ph.shape
        ))

    replay_act = tf.split(model.base_model(joint_obs_ph)[0], 2, axis=1)[0]
    novelty_loss = -norm(replay_act, peer_act_ph)  # only take

    policy.novelty_loss = novelty_loss

    alpha = policy.config["novelty_loss_param"]

    total_loss = (1 - alpha) * loss + alpha * novelty_loss
    policy.total_loss = total_loss
    return total_loss


def kl_and_loss_stats_modified(policy, train_batch):
    return {
        "novelty_loss": policy.novelty_loss,
        "cur_kl_coeff": tf.cast(policy.kl_coeff, tf.float64),
        "cur_lr": tf.cast(policy.cur_lr, tf.float64),
        # "total_loss": policy.loss_obj.loss,
        "total_loss": policy.total_loss,
        "policy_loss": policy.loss_obj.mean_policy_loss,
        "vf_loss": policy.loss_obj.mean_vf_loss,
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS],
            policy.model.value_function()),
        "kl": policy.loss_obj.mean_kl,
        "entropy": policy.loss_obj.mean_entropy,
        "entropy_coeff": tf.cast(policy.entropy_coeff, tf.float64),
    }


from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG

extra_loss_ppo_tf_policy_default_config = DEFAULT_CONFIG.copy()
extra_loss_ppo_tf_policy_default_config.update(
    novelty_loss_param=0.5, joint_dataset_sample_batch_size=200
)

AdditionalLossPPOTFPolicy = PPOTFPolicy.with_updates(
    get_default_config=lambda: extra_loss_ppo_tf_policy_default_config,
    postprocess_fn=postprocess_ppo_gae,
    stats_fn=kl_and_loss_stats_modified,
    loss_fn=ppo_surrogate_loss,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin, AddLossMixin
    ])


def process_multiagent_batch_fn(multi_agent_batch, self_optimizer):
    config = self_optimizer.workers._remote_config
    sample_size = config.get("joint_dataset_sample_batch_size")
    assert sample_size is not None, "You should specify the value of: " \
                                    "joint_dataset_sample_batch_size " \
                                    "in config!"

    print("[DEBUG] In process_mtag_batch_fn: sample_size is: ", sample_size)

    samples = [multi_agent_batch]
    cound_dict = defaultdict(
        int, {k: v.count for k, v in multi_agent_batch.policy_batches.items()}
    )
    while any([v < sample_size for v in cound_dict.values()]):
        tmp_batch = self_optimizer.workers.local_worker().sample()
        samples.append(tmp_batch)
        for k, v in tmp_batch.policy_batches.items():
            cound_dict[k] += v.count
    multi_agent_batch = MultiAgentBatch.concat_samples(samples)

    # FIXME: problem. nobody can promise the multi_agent_batch has sufficient \
    #  data.
    assert all([
        b.count >= sample_size for b in \
        multi_agent_batch.policy_batches.values()
    ]), [b.count for b in multi_agent_batch.policy_batches.values()]

    joint_obs = []
    for pid, batch in multi_agent_batch.policy_batches.items():
        count = batch.count
        batch.shuffle()
        if count < sample_size:
            logger.info("Your rollout sample size {} is "
                        "less than the replay sample size {}! "
                        "Check codes here!".format(count, sample_size))
            cnt = 0
            while True:
                end = min(count, sample_size - cnt)
                joint_obs.append(batch.slice(0, end)['obs'])
                if end < count:
                    break
                cnt += end
                batch.shuffle()
        else:
            joint_obs.append(batch.slice(0, sample_size)['obs'])
    joint_obs = np.concatenate(joint_obs)
    assert len(joint_obs) == sample_size * len(
        multi_agent_batch.policy_batches)

    def _replay(policy, pid):
        act, _, infos = policy.compute_actions(joint_obs)
        return pid, act, infos

    # TODO: can check if the infos[logits] equal to act.
    ret = {
        pid: act
        for pid, act, infos in
        self_optimizer.workers.local_worker().foreach_policy(_replay)
    }
    assert joint_obs.shape[0] % len(ret) == 0, (joint_obs.shape, len(ret))
    assert joint_obs.shape[0] == len(ret) * sample_size, (
        {k: v.shape for k, v in ret.items()}, joint_obs.shape, len(ret))
    return {JOINT_OBS: joint_obs, PEER_ACTION: ret}


def choose_policy_optimizer(workers, config):
    if config["simple_optimizer"]:
        return SyncSamplesOptimizer(
            workers,
            num_sgd_iter=config["num_sgd_iter"],
            train_batch_size=config["train_batch_size"],
            sgd_minibatch_size=config["sgd_minibatch_size"],
            standardize_fields=["advantages"])

    return LocalMultiGPUOptimizerModified(
        workers,
        [JOINT_OBS, PEER_ACTION],
        process_multiagent_batch_fn,
        sgd_batch_size=config["sgd_minibatch_size"],
        num_sgd_iter=config["num_sgd_iter"],
        num_gpus=config["num_gpus"],
        sample_batch_size=config["sample_batch_size"],
        num_envs_per_worker=config["num_envs_per_worker"],
        train_batch_size=config["train_batch_size"],
        standardize_fields=["advantages"],
        shuffle_sequences=config["shuffle_sequences"])


ExtraLossPPOTrainer = PPOTrainer.with_updates(
    default_config=extra_loss_ppo_tf_policy_default_config,
    default_policy=AdditionalLossPPOTFPolicy,
    make_policy_optimizer=choose_policy_optimizer
)

if __name__ == '__main__':
    from toolbox import initialize_ray
    from ray import tune
    from toolbox.utils import get_local_dir
    from toolbox.train.train_individual_marl import on_train_result

    # This is only test code.
    initialize_ray(test_mode=True, local_mode=False)
    num_agents = 5

    policy_names = ["ppo_agent{}".format(i) for i in range(num_agents)]

    env_config = {"env_name": "BipedalWalker-v2", "agent_ids": policy_names}
    env = MultiAgentEnvWrapper(env_config)
    config = {
        "env": MultiAgentEnvWrapper,
        "env_config": env_config,
        "log_level": "DEBUG",
        "joint_dataset_sample_batch_size": 131,
        "multiagent": {
            "policies": {i: (None, env.observation_space, env.action_space, {})
                         for i in policy_names},
            "policy_mapping_fn": lambda x: x,
        },
        "callbacks": {
            "on_train_result": on_train_result
        },
    }

    tune.run(
        ExtraLossPPOTrainer,
        local_dir=get_local_dir(),
        name="DELETEME_TEST_extra_loss_ppo_trainer",
        # checkpoint_at_end=True,
        # checkpoint_freq=10,
        stop={"timesteps_total": 10000},
        config=config,
    )
