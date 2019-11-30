import logging

import numpy as np
import tensorflow as tf
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.rllib.agents.ppo.ppo_policy import Postprocessing, SampleBatch, \
    BEHAVIOUR_LOGITS, ACTION_LOGP, postprocess_ppo_gae, PPOTFPolicy, \
    PPOLoss as OriginalPPOLoss, kl_and_loss_stats

from toolbox.marl.extra_loss_ppo_trainer import merge_dicts, \
    LocalMultiGPUOptimizerModified, chop_into_sequences, MultiAgentBatch, \
    SyncSamplesOptimizer, mixin_list, setup_mixins, validate_config

logger = logging.getLogger(__name__)

ceppo_default_config = merge_dicts(
    DEFAULT_CONFIG, dict(
        learn_with_peers=True,
        use_joint_dataset=False,
        disable=False
    )
)

required_peer_data_keys = {
    Postprocessing.VALUE_TARGETS, Postprocessing.ADVANTAGES,
    SampleBatch.ACTIONS, BEHAVIOUR_LOGITS, ACTION_LOGP, SampleBatch.VF_PREDS
}


class CEPPOLossObj:
    def __init__(self, loss_dict):
        self.loss = tf.reduce_mean(
            tf.stack([l.loss for l in loss_dict.values()]))
        self.mean_policy_loss = tf.reduce_mean(
            tf.stack([l.mean_policy_loss for l in loss_dict.values()]))
        self.mean_vf_loss = tf.reduce_mean(
            tf.stack([l.mean_vf_loss for l in loss_dict.values()]))
        self.mean_kl = tf.reduce_mean(
            tf.stack([l.mean_kl for l in loss_dict.values()]))
        self.mean_entropy = tf.reduce_mean(
            tf.stack([l.mean_entropy for l in loss_dict.values()]))


def ceppo_loss(policy, model, dist_class, train_batch):
    logits, state = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)

    if state:
        max_seq_len = tf.reduce_max(train_batch["seq_lens"])
        mask = tf.sequence_mask(train_batch["seq_lens"], max_seq_len)
        mask = tf.reshape(mask, [-1])
    else:
        mask = tf.ones_like(
            train_batch[Postprocessing.ADVANTAGES], dtype=tf.bool)

    other_names = set(
        k.split('-')[1] for k in train_batch.keys() if k.startswith("peer"))
    assert other_names or policy.config['disable']
    other_names.add("prev")
    policy.loss_dict = {}
    my_name = tf.get_variable_scope().name
    for peer in other_names:
        if peer == my_name:
            continue  # exclude myself.
        if policy.config['disable'] and peer != "prev":
            continue
        find = lambda x: x if peer == "prev" else "peer-{}-{}".format(peer, x)
        policy.loss_dict[peer] = OriginalPPOLoss(
            policy.action_space,
            dist_class,
            model,
            train_batch[find(Postprocessing.VALUE_TARGETS)],
            train_batch[find(Postprocessing.ADVANTAGES)],
            train_batch[find(SampleBatch.ACTIONS)],
            train_batch[find(BEHAVIOUR_LOGITS)],
            train_batch[find(ACTION_LOGP)],
            train_batch[find(SampleBatch.VF_PREDS)],
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
    if policy.config['disable']:
        assert len(policy.loss_dict) == 1
    if policy.config['learn_with_peers']:
        policy.loss_obj = CEPPOLossObj(policy.loss_dict)
    else:
        raise NotImplementedError("Haven't implement learn_with_peers==False")
    return policy.loss_obj.loss


class AddLossMixin(object):
    """When training, this Mixin add the sharing data into feed_dict.

    Originally, it's a function of tf_policy, so we copied some codes from
    tf_policy.py"""

    def _get_loss_inputs_dict(
            self, batch, shuffle, cross_policy_obj, policy_id=None
    ):
        """When training, add the required data into the feed_dict."""
        feed_dict = {}
        for pid, batch in cross_policy_obj.items():
            if pid == policy_id or self.config['disable']:
                continue
            for k in required_peer_data_keys:
                item_name = "peer-{}-{}".format(pid, k)
                feed_dict[self._loss_input_dict[item_name]] = batch[k]

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

        assert set(feed_dict.keys()) == set(self._loss_input_dict.keys()), \
            (set(feed_dict.keys()), set(self._loss_input_dict.keys()))
        return feed_dict


def postprocess_ceppo(policy, sample_batch, other_agent_batches=None,
                      epidose=None):
    batch = postprocess_ppo_gae(policy, sample_batch,
                                other_agent_batches, epidose)
    if not policy.loss_initialized():
        for key in required_peer_data_keys:
            for name in policy.config["multiagent"]["policies"].keys():
                assert not key.startswith("peer")
                batch["peer-{}-{}".format(name, key)] = \
                    np.zeros_like(batch[key], dtype=np.float32)
    return batch


def get_cross_policy_object(multi_agent_batch, self_optimizer):
    """Add contents into cross_policy_object, which passed to each policy."""
    count_dict = {k: v.count for k, v in
                  multi_agent_batch.policy_batches.items()}
    for k in self_optimizer.workers.local_worker().policy_map.keys():
        if k not in count_dict:
            count_dict[k] = 0
    sample_size = max(count_dict.values())
    ret = {}
    if min(count_dict.values()) < sample_size:
        samples = [multi_agent_batch]
        while min(count_dict.values()) < sample_size:
            tmp_batch = self_optimizer.workers.local_worker().sample()
            samples.append(tmp_batch)
            for k, b in tmp_batch.policy_batches.items():
                count_dict[k] += b.count
        multi_agent_batch = MultiAgentBatch.concat_samples(samples)
    for pid, batch in multi_agent_batch.policy_batches.items():
        batch.shuffle()
        ret[pid] = batch.slice(0, sample_size)
    assert 1 == len(set(b.count for b in ret.values())), ret
    return ret


def choose_policy_optimizer(workers, config):
    if config["simple_optimizer"]:
        return SyncSamplesOptimizer(
            workers,
            num_sgd_iter=config["num_sgd_iter"],
            train_batch_size=config["train_batch_size"],
            sgd_minibatch_size=config["sgd_minibatch_size"],
            standardize_fields=["advantages"]
        )
    return LocalMultiGPUOptimizerModified(
        workers, [], get_cross_policy_object,
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
    AddLossMixin.__init__(policy)
    setup_mixins(policy, obs_space, action_space, config)


def validate_config_modified(config):
    validate_config(config)


CEPPOTFPolicy = PPOTFPolicy.with_updates(
    name="CEPPOTFPolicy",
    get_default_config=lambda: ceppo_default_config,
    loss_fn=ceppo_loss,
    postprocess_fn=postprocess_ceppo,
    before_loss_init=setup_mixins_modified,
    stats_fn=kl_and_loss_stats,
    mixins=mixin_list + [AddLossMixin]
)

CEPPOTrainer = PPOTrainer.with_updates(
    name="CEPPO",
    default_config=ceppo_default_config,
    default_policy=CEPPOTFPolicy,
    validate_config=validate_config_modified,
    make_policy_optimizer=choose_policy_optimizer
)


def test_ceppo():
    from toolbox.marl.test_extra_loss import _base

    _base(CEPPOTrainer, False, extra_config={
        "learn_with_peers": True,
        "sample_batch_size": 200,
        "train_batch_size": 400,
        "sgd_minibatch_size": 128,
    }, t=50000, env_name="CartPole-v0")


if __name__ == '__main__':
    from ray import tune
    from toolbox import initialize_ray
    from toolbox.marl import MultiAgentEnvWrapper

    initialize_ray(test_mode=False)

    env_name = "CartPole-v0"
    num_agents = 3
    policy_names = ["ppo_agent{}".format(i) for i in range(num_agents)]
    env_config = {"env_name": env_name, "agent_ids": policy_names}
    env = MultiAgentEnvWrapper(env_config)
    config = {
        "env": MultiAgentEnvWrapper,
        "env_config": env_config,
        "multiagent": {
            "policies": {
                i: (None, env.observation_space, env.action_space, {})
                for i in policy_names
            },
            "policy_mapping_fn": lambda x: x,
        },
        "disable": True
    }

    tune.run(
        CEPPOTrainer,
        name="DELETEME_TEST_CEPPO",
        stop={"timesteps_total": 100000},
        config=config
    )
