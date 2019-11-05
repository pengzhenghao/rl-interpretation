import logging

import numpy as np
import tensorflow as tf
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy, PPOLoss, \
    LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin, ValueNetworkMixin
from ray.rllib.evaluation.postprocessing import compute_advantages, \
    Postprocessing
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import ACTION_LOGP

from toolbox import initialize_ray
from toolbox.marl import MultiAgentEnvWrapper

from ray.rllib.optimizers import SyncSamplesOptimizer
from toolbox.modified_rllib.multi_gpu_optimizer import LocalMultiGPUOptimizerModified
from ray.rllib.policy.sample_batch import MultiAgentBatch

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
    """This file is copied from ray.rllib"""
    """Adds the policy logits, VF preds, and advantages to the trajectory."""

    print('enter: postprocess_ppo_gae')

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
        batch[JOINT_OBS] = np.zeros_like(
            sample_batch[SampleBatch.CUR_OBS], dtype=np.float32)
        batch[PEER_ACTION] = np.zeros_like(
            sample_batch[SampleBatch.ACTIONS], dtype=np.float32)

    return batch
    # return batch
    # if policy.loss_initialized():
    #     pass
    #     if other_agent_batches and ("joint_dataset" not in
    #     episode.user_data):
    #         policy_names = list(episode._policies.keys())
    #         policy_names = sorted(
    #             policy_names,
    #             key=lambda ppo_agentx: int(ppo_agentx.split('agent')[1])
    #         )
    #
    #         tmp_dataset = {}
    #         for pid, (_, other_agent_batch) in other_agent_batches.items():
    #             assert pid in policy_names
    #             tmp_dataset[pid] = other_agent_batch[
    #             other_agent_batch.CUR_OBS]
    #
    #         # Because this postprocess function is in per-policy mode,
    #         # instead of in a cross-policy (multi-agent) mode,
    #         # so in fact I do not know what current policy name is!
    #         # I have to guess from the info I have now.
    #         my_name = set(policy_names).difference(set(tmp_dataset.keys()))
    #         assert len(my_name) == 1, (my_name, policy_names,
    #         other_agent_batches)
    #         my_name = my_name.pop()
    #
    #         tmp_dataset[my_name] = sample_batch[sample_batch.CUR_OBS]
    #
    #         assert set(policy_names) == set(tmp_dataset.keys())
    #
    #         joint_dataset = np.concatenate([
    #             tmp_dataset[pid] for pid in policy_names
    #         ])  # in fact it is joint observation
    #
    #         # batch['joint_dataset'] = joint_dataset
    #
    #         print("Finish build joint dataset its shape: {},"
    #               " the shape of each element: {}".format(
    #             joint_dataset.shape, [v.shape for v in tmp_dataset.values()]
    #         ))
    #
    #         # episode.user_data['joint_dataset'] = joint_dataset
    # else:
    #     print('strange. joint dataset already exist??',
    #           episode.user_data['joint_dataset'].shape)
    # assert (not policy.loss_initialized()) or \
    #        ("joint_dataset" in episode.user_data)
    #
    # act, _, infos = policy.compute_actions(episode.user_data[
    # 'joint_dataset'])
    # else:
    #     batch['joint_dataset'] = np.zeros_like(
    #         sample_batch[SampleBatch.CUR_OBS], dtype=np.float32)
    #     # make a fake action
    #     # act = sample_batch[SampleBatch.ACTIONS]
    #     act = np.zeros_like(
    #         sample_batch[SampleBatch.ACTIONS])
    #
    # batch["joint_dataset_replay"] = act
    # return batch


from ray.rllib.policy.rnn_sequencing import chop_into_sequences


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

        print("please stop here", cross_policy_obj)

        feed_dict = {}
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

    print('enter: ppo_surrogate_loss')

    # if policy.loss_initialized():
    # print('please stop here loss init?: ', policy.loss_initialized())

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

    # just for test
    # loss += tf.norm(train_batch[JOINT_OBS])

    joint_obs_ph = train_batch[JOINT_OBS]
    joint_obs_length = joint_obs_ph.shape.as_list()[0]
    peer_act_ph = train_batch[PEER_ACTION]

    # test only
    # peer_act_ph_shape = peer_act_ph.shape.as_list()
    # assert len(peer_act_ph_shape) == 2
    # assert peer_act_ph_shape[0] % joint_obs_length == 0

    def reshape(x):
        return x
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

    replay_act = model.base_model(joint_obs_ph)[0][:, :4]
    novelty_loss = norm(replay_act, reshape(peer_act_ph))

    return (loss + novelty_loss) / 2


AdditionalLossPPOTFPolicy = PPOTFPolicy.with_updates(
    postprocess_fn=postprocess_ppo_gae,
    loss_fn=ppo_surrogate_loss,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin, AddLossMixin
    ])

def process_multiagent_batch_fn(multiagent_batch):
    assert isinstance(multiagent_batch, MultiAgentBatch)
    print("PSH!!!")




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


def on_sample_end(info):
    print('please stop here')

    worker = info['worker']
    multi_agent_batch = info['samples']
    sample_size = 200

    # worker = trainer.workers.local_worker()
    # joint_obs = _collect_joint_dataset(trainer, worker, sample_size)

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

    print("joint_obs shape: {}, policy sample sizes: {}".format(
        joint_obs.shape,
        [b.count for b in multi_agent_batch.policy_batches.values()]
    ))

    # print('pls stop here')

    def _replay(policy, pid):
        act, _, infos = policy.compute_actions(joint_obs)
        return pid, act, infos

    ret = [
        act
        for pid, act, infos in worker.foreach_policy(_replay)
    ]

    ret = np.stack(ret)

    # for batch in multi_agent_batch.policy_batches.values():
    #     batch[JOINT_OBS] = joint_obs
    #     batch[PEER_ACTION] = ret

    print("please stop hrer")

    # # now we have a mapping: policy_id to joint_dataset_replay in 'ret'
    #
    # flatten = [act for act, infos in ret.values()]  # flatten action array
    # dist_matrix = joint_dataset_distance(flatten)


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
        "callbacks": {
            "on_sample_end": on_sample_end
        }
    }

    agent = AdditionalLossPPOTrainer(
        env=MultiAgentEnvWrapper,
        config=config
    )

    agent.train()
    # agent.train()
    # agent.train()
