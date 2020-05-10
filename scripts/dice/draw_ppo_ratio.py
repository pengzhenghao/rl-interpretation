from ray.rllib.agents.ppo.ppo import PPOTFPolicy, PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import kl_and_loss_stats, tf, \
    Postprocessing, SampleBatch

from toolbox.train import train, get_train_parser


class PPOLoss:
    def __init__(
            self, dist_class, model, value_targets, advantages, actions,
            prev_logits, prev_actions_logp, vf_preds, curr_action_dist,
            value_fn, cur_kl_coeff, valid_mask, entropy_coeff=0, clip_param=0.1,
            vf_clip_param=0.1, vf_loss_coeff=1.0, use_gae=True
    ):
        if valid_mask is not None:
            def reduce_mean_valid(t):
                return tf.reduce_mean(tf.boolean_mask(t, valid_mask))
        else:
            def reduce_mean_valid(t):
                return tf.reduce_mean(t)
        prev_dist = dist_class(prev_logits, model)
        # Make loss functions.
        logp_ratio = tf.exp(curr_action_dist.logp(actions) - prev_actions_logp)
        action_kl = prev_dist.kl(curr_action_dist)
        self.mean_kl = reduce_mean_valid(action_kl)
        self.debug_ratio = logp_ratio  # <<== Here!

        curr_entropy = curr_action_dist.entropy()
        self.mean_entropy = reduce_mean_valid(curr_entropy)

        surrogate_loss = tf.minimum(
            advantages * logp_ratio,
            advantages * tf.clip_by_value(logp_ratio, 1 - clip_param,
                                          1 + clip_param))
        self.mean_policy_loss = reduce_mean_valid(-surrogate_loss)

        if use_gae:
            vf_loss1 = tf.square(value_fn - value_targets)
            vf_clipped = vf_preds + tf.clip_by_value(
                value_fn - vf_preds, -vf_clip_param, vf_clip_param)
            vf_loss2 = tf.square(vf_clipped - value_targets)
            vf_loss = tf.maximum(vf_loss1, vf_loss2)
            self.mean_vf_loss = reduce_mean_valid(vf_loss)
            loss = reduce_mean_valid(
                -surrogate_loss + cur_kl_coeff * action_kl +
                vf_loss_coeff * vf_loss - entropy_coeff * curr_entropy)
        else:
            self.mean_vf_loss = tf.constant(0.0)
            loss = reduce_mean_valid(-surrogate_loss +
                                     cur_kl_coeff * action_kl -
                                     entropy_coeff * curr_entropy)
        self.loss = loss


def ppo_surrogate_loss(policy, model, dist_class, train_batch):
    logits, state = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)
    mask = None
    if state:
        max_seq_len = tf.reduce_max(train_batch["seq_lens"])
        mask = tf.sequence_mask(train_batch["seq_lens"], max_seq_len)
        mask = tf.reshape(mask, [-1])
    policy.loss_obj = PPOLoss(
        dist_class,
        model,
        train_batch[Postprocessing.VALUE_TARGETS],
        train_batch[Postprocessing.ADVANTAGES],
        train_batch[SampleBatch.ACTIONS],
        train_batch[SampleBatch.ACTION_DIST_INPUTS],
        train_batch[SampleBatch.ACTION_LOGP],
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
    )
    return policy.loss_obj.loss


def kl_and_loss_stats_new(policy, train_batch):
    ret = kl_and_loss_stats(policy, train_batch)
    ret["debug_ratio"] = policy.loss_obj.debug_ratio
    return ret


PPORatioPolicy = PPOTFPolicy.with_updates(
    name="PPORatioPolicy",
    loss_fn=ppo_surrogate_loss,
    stats_fn=kl_and_loss_stats_new,
)

PPORatioTrainer = PPOTrainer.with_updates(
    name="PPORatioTrainer",
    default_policy=PPORatioPolicy,
    get_policy_class=lambda _: PPORatioPolicy
)

if __name__ == '__main__':
    parser = get_train_parser()
    args = parser.parse_args()

    env_name = "Walker2d-v3"
    exp_name = "{}-{}".format(args.exp_name, env_name)
    large = True
    stop = int(5e7)

    config = {
        "env": "Walker2d-v3",
        "kl_coeff": 1.0,
        "num_sgd_iter": 10,
        "lr": 0.0001,
        'rollout_fragment_length': 200 if large else 50,
        'sgd_minibatch_size': 100 if large else 64,
        'train_batch_size': 10000 if large else 2048,
        "num_gpus": 0.4,
        "num_cpus_per_worker": args.num_cpus_per_worker,
        "num_cpus_for_driver": args.num_cpus_for_driver,
        "num_envs_per_worker": 8 if large else 10,
        'num_workers': 8 if large else 1,
        "callbacks": {"on_train_result": None}
    }

    train(
        PPORatioTrainer,
        config=config,
        stop=stop,
        exp_name=exp_name,
        num_seeds=args.num_seeds,
        num_gpus=args.num_gpus,
        test_mode=args.test,
        keep_checkpoints_num=10
    )
