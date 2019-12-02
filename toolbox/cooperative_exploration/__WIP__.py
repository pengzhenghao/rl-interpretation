import tensorflow as tf

from ray.rllib.agents.ppo.ppo_policy import PPOLoss as OriginalPPOLoss, \
    SampleBatch, BEHAVIOUR_LOGITS, ACTION_LOGP, Postprocessing


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
