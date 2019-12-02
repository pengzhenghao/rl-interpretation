import logging

import tensorflow as tf
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.rllib.agents.ppo.ppo_policy import Postprocessing, SampleBatch, \
    BEHAVIOUR_LOGITS, ACTION_LOGP, postprocess_ppo_gae, PPOTFPolicy, \
    PPOLoss as OriginalPPOLoss, setup_mixins, make_tf_callable

from toolbox.marl.extra_loss_ppo_trainer import merge_dicts, \
    mixin_list

logger = logging.getLogger(__name__)

ceppo_default_config = merge_dicts(
    DEFAULT_CONFIG, dict(
        learn_with_peers=True,
        use_myself_vf_preds=True,
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

        if state:
            raise NotImplementedError()
            # max_seq_len = tf.reduce_max(train_batch["seq_lens"])
            # mask = tf.sequence_mask(train_batch["seq_lens"], max_seq_len)
            # mask = tf.reshape(mask, [-1])
        else:
            mask = tf.ones_like(train_batch[find(Postprocessing.ADVANTAGES)],
                                dtype=tf.bool)

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


def postprocess_ceppo(policy, sample_batch, others_batches=None, epidose=None):
    if not policy.loss_initialized() or policy.config['disable']:
        # Only for initialization. GAE postprocess done in the following func.
        return postprocess_ppo_gae(policy, sample_batch)

    if not policy.config["use_myself_vf_preds"]:
        batch = SampleBatch.concat_samples(
            [sample_batch] + [b for (_, b) in others_batches.values()])
        return postprocess_ppo_gae(policy, batch)

    # use_myself_vf_preds
    assert policy.config["use_myself_vf_preds"]
    batches = [postprocess_ppo_gae(policy, sample_batch)]
    for pid, (_, batch) in others_batches.items():
        batch[SampleBatch.VF_PREDS] = policy._value_batch(
            batch[SampleBatch.CUR_OBS], batch[SampleBatch.PREV_ACTIONS],
            batch[SampleBatch.PREV_REWARDS])
        # use my policy to postprocess other's trajectory.
        batches.append(postprocess_ppo_gae(policy, batch))
    return SampleBatch.concat_samples(batches)


class ValueNetworkMixin2(object):
    def __init__(self, config):
        self._update_value_batch_function(config['use_gae'])

    def _update_value_batch_function(self, use_gae=None):
        if use_gae is None:
            use_gae = self.config['use_gae']
        if use_gae:
            @make_tf_callable(self.get_session(), True)
            def value_batch(ob, prev_action, prev_reward):
                # We do not support recurrent network now.
                model_out, _ = self.model({
                    SampleBatch.CUR_OBS: tf.convert_to_tensor(ob),
                    SampleBatch.PREV_ACTIONS: tf.convert_to_tensor(
                        prev_action),
                    SampleBatch.PREV_REWARDS: tf.convert_to_tensor(
                        prev_reward),
                    "is_training": tf.convert_to_tensor(False),
                })
                return self.model.value_function()
        else:
            @make_tf_callable(self.get_session(), True)
            def value_batch(ob, prev_action, prev_reward):
                return tf.zeros_like(prev_reward)

        self._value_batch = value_batch


def setup_mixins_modified(policy, obs_space, action_space, config):
    ValueNetworkMixin2.__init__(policy, config)
    setup_mixins(policy, obs_space, action_space, config)


CEPPOTFPolicy = PPOTFPolicy.with_updates(
    name="CEPPOTFPolicy",
    get_default_config=lambda: ceppo_default_config,
    postprocess_fn=postprocess_ceppo,
    before_loss_init=setup_mixins_modified,
    mixins=mixin_list + [ValueNetworkMixin2]
)

CEPPOTrainer = PPOTrainer.with_updates(
    name="CEPPO",
    default_config=ceppo_default_config,
    default_policy=CEPPOTFPolicy
)


def debug_ceppo(local_mode):
    from toolbox.marl.test_extra_loss import _base

    _base(CEPPOTrainer, local_mode, extra_config={
        "disable": True
    }, env_name="CartPole-v0")


def validate_ceppo(disable, test_mode=False):
    from ray import tune
    from toolbox import initialize_ray
    from toolbox.marl import MultiAgentEnvWrapper

    initialize_ray(test_mode=test_mode, local_mode=False)

    env_name = "CartPole-v0"
    num_agents = 3
    policy_names = ["ppo_agent{}".format(i) for i in range(num_agents)]
    env_config = {"env_name": env_name, "agent_ids": policy_names}
    env = MultiAgentEnvWrapper(env_config)
    config = {
        "seed": 0,
        "env": MultiAgentEnvWrapper,
        "env_config": env_config,
        "multiagent": {
            "policies": {
                i: (None, env.observation_space, env.action_space, {})
                for i in policy_names
            },
            "policy_mapping_fn": lambda x: x,
        },
        "disable": disable,
    }

    if disable:
        config['train_batch_size'] = \
            ceppo_default_config['train_batch_size'] * num_agents
        config['num_workers'] = \
            ceppo_default_config['num_workers'] * num_agents

    tune.run(
        CEPPOTrainer,
        name="DELETEME_TEST_CEPPO",
        # stop={"timesteps_total": 50000},
        stop={"info/num_steps_trained": 50000},
        config=config
    )


if __name__ == '__main__':
    # debug_ceppo(local_mode=False)
    validate_ceppo(disable=False, test_mode=True)
