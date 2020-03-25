import gym
import numpy as np
from ray import tune
from ray.rllib.agents.impala import vtrace
from ray.rllib.agents.impala.impala import ImpalaTrainer, VTraceTFPolicy
from ray.rllib.agents.impala.vtrace_policy import _make_time_major
from ray.rllib.models.tf.tf_action_dist import Categorical
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import ACTION_LOGP
from ray.rllib.utils import try_import_tf

BEHAVIOUR_LOGITS = "behaviour_logits"
tf = try_import_tf()


class VTraceLossModified:
    def __init__(self,
                 actions,
                 actions_logp,
                 actions_entropy,
                 dones,
                 behaviour_action_logp,
                 behaviour_logits,
                 target_logits,
                 discount,
                 rewards,
                 values,
                 bootstrap_value,
                 dist_class,
                 model,
                 valid_mask,
                 config,
                 vf_loss_coeff=0.5,
                 entropy_coeff=0.01,
                 clip_rho_threshold=1.0,
                 clip_pg_rho_threshold=1.0):
        # Compute vtrace on the CPU for better perf.
        with tf.device("/cpu:0"):
            self.vtrace_returns = vtrace.multi_from_logits(
                behaviour_action_log_probs=behaviour_action_logp,
                behaviour_policy_logits=behaviour_logits,
                target_policy_logits=target_logits,
                actions=tf.unstack(actions, axis=2),
                discounts=tf.to_float(~dones) * discount,
                rewards=rewards,
                values=values,
                bootstrap_value=bootstrap_value,
                dist_class=dist_class,
                model=model,
                clip_rho_threshold=tf.cast(clip_rho_threshold, tf.float32),
                clip_pg_rho_threshold=tf.cast(clip_pg_rho_threshold,
                                              tf.float32))
            self.value_targets = self.vtrace_returns.vs

            advantages = self.vtrace_returns.pg_advantages
            # The advantages has shape [Sample batch size, B] (B is the
            # number of sample_batch in train_batch). Here we normalize
            # advantages among whole train batch.
            advantages = (advantages - tf.reduce_mean(advantages)) / \
                         tf.maximum(1e-4, tf.math.reduce_std(advantages))

        # The policy gradients loss
        self.pi_loss = -tf.reduce_sum(
            tf.boolean_mask(actions_logp * advantages, valid_mask)
        )

        # The baseline loss
        delta = tf.boolean_mask(values - self.vtrace_returns.vs, valid_mask)
        self.vf_loss = 0.5 * tf.reduce_sum(tf.square(delta))

        # The entropy loss
        self.entropy = tf.reduce_sum(
            tf.boolean_mask(actions_entropy, valid_mask))

        # The summed weighted loss
        self.total_loss = (self.pi_loss + self.vf_loss * vf_loss_coeff -
                           self.entropy * entropy_coeff)


def build_vtrace_loss_modified(policy, model, dist_class, train_batch):
    model_out, _ = model.from_batch(train_batch)
    action_dist = dist_class(model_out, model)

    if isinstance(policy.action_space, gym.spaces.Discrete):
        is_multidiscrete = False
        output_hidden_shape = [policy.action_space.n]
    elif isinstance(policy.action_space,
                    gym.spaces.multi_discrete.MultiDiscrete):
        is_multidiscrete = True
        output_hidden_shape = policy.action_space.nvec.astype(np.int32)
    else:
        is_multidiscrete = False
        output_hidden_shape = 1

    def make_time_major(*args, **kw):
        return _make_time_major(policy, train_batch.get("seq_lens"), *args,
                                **kw)

    actions = train_batch[SampleBatch.ACTIONS]
    dones = train_batch[SampleBatch.DONES]
    rewards = train_batch[SampleBatch.REWARDS]
    behaviour_action_logp = train_batch[ACTION_LOGP]
    behaviour_logits = train_batch[BEHAVIOUR_LOGITS]
    unpacked_behaviour_logits = tf.split(
        behaviour_logits, output_hidden_shape, axis=1)
    unpacked_outputs = tf.split(model_out, output_hidden_shape, axis=1)
    values = model.value_function()

    if policy.is_recurrent():
        max_seq_len = tf.reduce_max(train_batch["seq_lens"]) - 1
        mask = tf.sequence_mask(train_batch["seq_lens"], max_seq_len)
        mask = tf.reshape(mask, [-1])
    else:
        mask = tf.ones_like(rewards)

    # Prepare actions for loss
    loss_actions = actions if is_multidiscrete else tf.expand_dims(
        actions, axis=1)

    # Inputs are reshaped from [B * T] => [T - 1, B] for V-trace calc.
    policy.loss = VTraceLossModified(
        actions=make_time_major(loss_actions, drop_last=True),
        actions_logp=make_time_major(
            action_dist.logp(actions), drop_last=True),
        actions_entropy=make_time_major(
            action_dist.multi_entropy(), drop_last=True),
        dones=make_time_major(dones, drop_last=True),
        behaviour_action_logp=make_time_major(
            behaviour_action_logp, drop_last=True),
        behaviour_logits=make_time_major(
            unpacked_behaviour_logits, drop_last=True),
        target_logits=make_time_major(unpacked_outputs, drop_last=True),
        discount=policy.config["gamma"],
        rewards=make_time_major(rewards, drop_last=True),
        values=make_time_major(values, drop_last=True),
        bootstrap_value=make_time_major(values)[-1],
        dist_class=Categorical if is_multidiscrete else dist_class,
        model=model,
        valid_mask=make_time_major(mask, drop_last=True),
        config=policy.config,
        vf_loss_coeff=policy.config["vf_loss_coeff"],
        entropy_coeff=policy.entropy_coeff,
        clip_rho_threshold=policy.config["vtrace_clip_rho_threshold"],
        clip_pg_rho_threshold=policy.config["vtrace_clip_pg_rho_threshold"])

    return policy.loss.total_loss


ANVTraceTFPolicy = VTraceTFPolicy.with_updates(
    name="ANVTraceTFPolicy",
    loss_fn=build_vtrace_loss_modified
)


def choose_policy(config):
    if config["vtrace"]:
        return ANVTraceTFPolicy
    else:
        raise NotImplementedError()


ANIMPALATrainer = ImpalaTrainer.with_updates(
    name="ANIMPALA",
    get_policy_class=choose_policy,
    default_policy=VTraceLossModified
)

if __name__ == '__main__':
    import yaml
    import argparse
    from toolbox import initialize_ray

    initialize_ray(test_mode=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", type=str, default="")
    parser.add_argument("--num-samples", type=int, default=5)
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {"env": "BipedalWalker-v2", "num_gpus": 0}

    tune.run(
        ANIMPALATrainer,
        config=config,
        num_samples=args.num_samples
    )
