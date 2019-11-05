import logging

import numpy as np
import tensorflow as tf
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy, PPOLoss, \
    LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin, ValueNetworkMixin
from ray.rllib.evaluation.postprocessing import compute_advantages, \
    Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import ACTION_LOGP

from ray.rllib.policy.rnn_sequencing import chop_into_sequences

from toolbox import initialize_ray
from toolbox.marl import MultiAgentEnvWrapper

from ray.rllib.optimizers import SyncSamplesOptimizer
from toolbox.modified_rllib.multi_gpu_optimizer import LocalMultiGPUOptimizerModified

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
        # add some variable in batch to initialize the place holder.
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

        print("pls")

        feed_dict = {}

        ########## parse the cross-policy info and put in feed_dict ##########
        joint_obs_ph = self._loss_input_dict[JOINT_OBS]
        feed_dict[joint_obs_ph] = cross_policy_obj[JOINT_OBS]

        replay_ph = self._loss_input_dict[PEER_ACTION]
        feed_dict[replay_ph] = np.concatenate(
            list(cross_policy_obj[PEER_ACTION].values())
        )
        ########## Finish add cross-policy info ##########

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

    joint_obs_ph = train_batch[JOINT_OBS]
    peer_act_ph = train_batch[PEER_ACTION]

    def reshape(obs, act):
        return tf.reshape(
            act,
            [-1, tf.shape(obs)[0], tf.shape(act)[1]]
        )
        # return x
        # act_shape = x.shape.as_list()[-1]
        # shape = [-1, joint_obs_length, act_shape]
        # return tf.reshape(x, shape)

    def norm(x, y):
        subtract = tf.subtract(x, y)
        norm = tf.reduce_mean(subtract ** 2)
        print(
            "Inside norm(x, y). the shape of sub: {}, the shape of norm "
            "{}".format(
                subtract.shape, norm.shape
            ), x.shape, y.shape)
        return norm

    replay_act = tf.split(model.base_model(joint_obs_ph)[0], 2, axis=1)[0]
    novelty_loss = -norm(replay_act, reshape(joint_obs_ph, peer_act_ph))

    return (loss + novelty_loss) / 2


AdditionalLossPPOTFPolicy = PPOTFPolicy.with_updates(
    postprocess_fn=postprocess_ppo_gae,
    loss_fn=ppo_surrogate_loss,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin, AddLossMixin
    ])


def process_multiagent_batch_fn(multi_agent_batch, workers):
    config = workers._remote_config
    sample_size = config.get("sample_batch_size") or 200

    joint_obs = []
    for pid, batch in multi_agent_batch.policy_batches.items():
        count = batch.count
        batch.shuffle()
        if count < sample_size:
            print("[WARNING]!!! Your rollout sample size is "
                  "less than the replay sample size! "
                  "Check codes here!")
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

    print("[JOINT_DATASET] joint_obs shape: {}, policy sample sizes: {}".format(
        joint_obs.shape,
        [b.count for b in multi_agent_batch.policy_batches.values()]
    ))

    def _replay(policy, pid):
        act, _, infos = policy.compute_actions(joint_obs)
        return pid, act, infos
    ret = {
        pid: act
        for pid, act, infos in workers.local_worker().foreach_policy(_replay)
    }
    info = {JOINT_OBS: joint_obs, PEER_ACTION: ret}
    return info


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


AdditionalLossPPOTrainer = PPOTrainer.with_updates(
    default_policy=AdditionalLossPPOTFPolicy,
    make_policy_optimizer=choose_policy_optimizer
)


if __name__ == '__main__':
    initialize_ray(test_mode=True, local_mode=True)
    num_agents = 2

    policy_names = ["ppo_agent{}".format(i) for i in range(num_agents)]

    env_config = {"env_name": "BipedalWalker-v2", "agent_ids": policy_names}
    env = MultiAgentEnvWrapper(env_config)
    config = {
        "env": MultiAgentEnvWrapper,
        "env_config": env_config,
        "log_level": "DEBUG",
        "sample_batch_size": 200,
        "multiagent": {
            "policies": {i: (None, env.observation_space, env.action_space, {})
                         for i in policy_names},
            "policy_mapping_fn": lambda x: x,
        },
        # "callbacks": {
        #     "on_sample_end": on_sample_end
        # }
    }

    agent = AdditionalLossPPOTrainer(
        env=MultiAgentEnvWrapper,
        config=config
    )

    agent.train()
    # agent.train()
    # agent.train()
