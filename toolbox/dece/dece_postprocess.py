from ray.rllib.agents.ppo.ppo_policy import postprocess_ppo_gae, ACTION_LOGP, \
    SampleBatch, BEHAVIOUR_LOGITS
from ray.rllib.evaluation.postprocessing import discount, Postprocessing
from ray.rllib.policy.tf_policy import ACTION_PROB

from toolbox.dece.utils import *
from toolbox.distance import get_kl_divergence


def postprocess_ppo_gae_replay(policy, batch, other_policy):
    """Adds the policy logits, VF preds, and advantages to the trajectory."""
    completed = batch["dones"][-1]
    if completed:
        my_last_r = other_last_r = 0.0
    else:
        next_state = []
        for i in range(policy.num_state_tensors()):
            next_state.append([batch["state_out_{}".format(i)][-1]])
        other_last_r = other_policy._value(
            batch[SampleBatch.NEXT_OBS][-1], batch[SampleBatch.ACTIONS][-1],
            batch[SampleBatch.REWARDS][-1], *next_state
        )
        my_last_r = policy._value(
            batch[SampleBatch.NEXT_OBS][-1], batch[SampleBatch.ACTIONS][-1],
            batch[SampleBatch.REWARDS][-1], *next_state
        )
    batch = compute_advantages_replay(
        batch,
        my_last_r,
        other_last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"]
    )
    return batch


def compute_advantages_replay(
        rollout, my_last_r, other_last_r, gamma=0.9, lambda_=1.0, use_gae=True
):
    """Given a rollout, compute its value targets and the advantage.

    Args:
        rollout (SampleBatch): SampleBatch of a single trajectory
        last_r (float): Value estimation for last observation
        gamma (float): Discount factor.
        lambda_ (float): Parameter for GAE
        use_gae (bool): Using Generalized Advantage Estimation

    Returns:
        SampleBatch (SampleBatch): Object with experience from rollout and
            processed rewards.
    """
    traj = {}
    trajsize = len(rollout[SampleBatch.ACTIONS])
    for key in rollout:
        traj[key] = np.stack(rollout[key])

    if use_gae:
        assert SampleBatch.VF_PREDS in rollout, "Values not found!"
        # vpred_t = np.concatenate(
        #     [rollout[SampleBatch.VF_PREDS],
        #      np.array([last_r])]
        # )
        # delta_t = \
        #     traj[SampleBatch.REWARDS] + gamma * vpred_t[1:] * (
        #             1 - lambda_) - vpred_t[:-1]

        # use other's values to compute advantages
        # this advantage will not be used to comput value target.
        other_vpred_t = np.concatenate(
            [rollout["other_vf_preds"],
             np.array([other_last_r])]
        )
        other_delta_t = (
            traj[SampleBatch.REWARDS] + gamma * other_vpred_t[1:] -
            other_vpred_t[:-1]
        )
        other_advantage = discount(other_delta_t, gamma * lambda_)
        other_advantage = (other_advantage - other_advantage.mean()
                           ) / max(1e-4, other_advantage.std())

        # we put other's advantage in 'advantages' field. We need to make sure
        # this field is not used in future postprocessing.
        traj[Postprocessing.ADVANTAGES] = other_advantage

        # Ratio is almost deprecated. We only use it to compute value target.
        ratio = np.exp(traj['action_logp'] - traj["other_action_logp"])
        # ratio = np.clip(ratio, 0.0, clip_action_prob_ratio)
        traj["debug_ratio"] = ratio
        fake_delta = np.zeros_like(ratio)
        fake_delta[-1] = 1
        traj["debug_fake_adv"] = calculate_gae_advantage(
            np.zeros_like(ratio), fake_delta, ratio, lambda_, gamma
        )

        # value_target = (
        #         traj[Postprocessing.ADVANTAGES] + traj[SampleBatch.VF_PREDS]
        # ).copy().astype(np.float32)
        my_vpred_t = np.concatenate(
            [rollout[SampleBatch.VF_PREDS],
             np.array([my_last_r])]
        )
        assert ratio.shape == traj[SampleBatch.REWARDS].shape
        # value_target = \
        #     ratio * (traj[SampleBatch.REWARDS] + gamma * my_vpred_t[1:])

        value_target = (
            ratio * (traj[SampleBatch.REWARDS] + gamma * my_vpred_t[1:]) +
            (1 - ratio) * (my_vpred_t[:-1])
        )

        traj[Postprocessing.VALUE_TARGETS] = value_target

    else:
        raise NotImplementedError()
        # rewards_plus_v = np.concatenate(
        #     [rollout[SampleBatch.REWARDS],
        #      np.array([last_r])])
        #
        # ratio = np.exp(traj['action_logp'] - traj["other_action_logp"])
        # assert_nan(ratio)
        # delta = discount(rewards_plus_v, gamma)[:-1]
        #
        # assert delta.shape == ratio.shape
        #
        # traj[Postprocessing.ADVANTAGES] = ratio * delta
        # traj[Postprocessing.VALUE_TARGETS] = np.zeros_like(
        #     traj[Postprocessing.ADVANTAGES])

    traj[Postprocessing.ADVANTAGES
         ] = traj[Postprocessing.ADVANTAGES].copy().astype(np.float32)

    assert all(val.shape[0] == trajsize for val in traj.values()), \
        "Rollout stacked incorrectly!"
    return SampleBatch(traj)


def calculate_gae_advantage(values, delta, ratio, lambda_, gamma):
    assert len(delta) > 0, (values, delta)
    assert values.shape == delta.shape
    y_n = np.zeros_like(delta)
    y_n[-1] = delta[-1]
    length = len(delta)
    for ind in range(length - 2, -1, -1):
        # ind = 8, 7, 6, ..., 0 if length = 10
        y_n[ind] = delta[ind] + ratio[
            ind + 1] * gamma * lambda_ * (y_n[ind + 1] + values[ind + 1])
    return y_n


def _compute_logp(logit, x):
    # Only for DiagGaussian distribution. Copied from tf_action_dist.py
    logit = logit.astype(np.float64)
    x = np.expand_dims(x.astype(np.float64), 1) if x.ndim == 1 else x
    mean, log_std = np.split(logit, 2, axis=1)
    logp = (
        -0.5 * np.sum(np.square((x - mean) / np.exp(log_std)), axis=1) -
        0.5 * np.log(2.0 * np.pi) * x.shape[1] - np.sum(log_std, axis=1)
    )
    p = np.exp(logp)
    return logp, p


def _clip_batch(other_batch, clip_action_prob_kl):
    kl = get_kl_divergence(
        source=other_batch[BEHAVIOUR_LOGITS],
        target=other_batch["other_logits"],
        mean=False
    )

    mask = kl < clip_action_prob_kl
    length = len(mask)
    info = {"kl": kl, "unclip_length": length, "length": len(mask)}

    if not np.all(mask):
        length = mask.argmin()
        info['unclip_length'] = length
        if length == 0:
            return None, info
        assert length < len(other_batch['action_logp'])
        other_batch = other_batch.slice(0, length)

    return other_batch, info


def postprocess_diversity(policy, batch, others_batches, episode=None):
    # if policy.config[DIVERSITY_ENCOURAGING]:
    completed = batch["dones"][-1]
    batch[NOVELTY_REWARDS] = policy.compute_novelty(
        batch, others_batches, episode
    )

    if completed:
        last_r_novelty = 0.0
    else:
        next_state = []
        for i in range(policy.num_state_tensors()):
            next_state.append([batch["state_out_{}".format(i)][-1]])
        last_r_novelty = policy._novelty_value(
            batch[SampleBatch.NEXT_OBS][-1], batch[SampleBatch.ACTIONS][-1],
            batch[NOVELTY_REWARDS][-1], *next_state
        )

    # compute the advantages of novelty rewards
    novelty_advantages, novelty_value_target = \
        _compute_advantages_for_diversity(
            rewards=batch[NOVELTY_REWARDS],
            last_r=last_r_novelty,
            gamma=policy.config["gamma"],
            lambda_=policy.config["lambda"],
            values=batch[NOVELTY_VALUES]
            if policy.config[USE_DIVERSITY_VALUE_NETWORK] else None,
            use_gae=policy.config[USE_DIVERSITY_VALUE_NETWORK]
        )
    batch[NOVELTY_ADVANTAGES] = novelty_advantages
    batch[NOVELTY_VALUE_TARGETS] = novelty_value_target
    return batch


def postprocess_dece(policy, sample_batch, others_batches=None, episode=None):
    if not policy.loss_initialized():
        batch = postprocess_ppo_gae(policy, sample_batch)
        batch["advantages_unnormalized"] = np.zeros_like(
            batch["advantages"], dtype=np.float32
        )
        batch['debug_ratio'] = np.zeros_like(
            batch["advantages"], dtype=np.float32
        )
        batch['debug_fake_adv'] = np.zeros_like(
            batch["advantages"], dtype=np.float32
        )
        if policy.config[DIVERSITY_ENCOURAGING]:
            batch[NOVELTY_REWARDS] = np.zeros_like(
                batch["advantages"], dtype=np.float32
            )
            batch[NOVELTY_VALUE_TARGETS] = np.zeros_like(
                batch["advantages"], dtype=np.float32
            )
            batch[NOVELTY_ADVANTAGES] = np.zeros_like(
                batch["advantages"], dtype=np.float32
            )
        return batch

    batch = sample_batch
    if policy.config[REPLAY_VALUES]:
        # a little workaround. We normalize advantage for all batch before
        # concatnation.
        tmp_batch = postprocess_ppo_gae(policy, batch)
        tmp_batch = postprocess_diversity(policy, tmp_batch, others_batches)
        value = tmp_batch[Postprocessing.ADVANTAGES]
        standardized = (value - value.mean()) / max(1e-4, value.std())
        tmp_batch[Postprocessing.ADVANTAGES] = standardized
        batches = [tmp_batch]
    else:
        batch = postprocess_ppo_gae(policy, batch)
        batch = postprocess_diversity(policy, batch, others_batches)
        batches = [batch]

    for pid, (other_policy, other_batch_raw) in others_batches.items():
        # The logic is that EVEN though we may use DISABLE or NO_REPLAY_VALUES,
        # but we still want to take a look of those statics.
        # Maybe in the future we can add knob to remove all such slowly stats.

        if other_batch_raw is None:
            continue

        other_batch_raw = other_batch_raw.copy()
        other_batch = other_batch_raw.copy()

        # four fields that we will overwrite.
        # Two additional: advantages / value target
        other_batch["other_action_logp"] = other_batch[ACTION_LOGP].copy()
        other_batch["other_action_prob"] = other_batch[ACTION_PROB].copy()
        other_batch["other_logits"] = other_batch[BEHAVIOUR_LOGITS].copy()
        other_batch["other_vf_preds"] = other_batch[SampleBatch.VF_PREDS
                                                    ].copy()

        # use my policy to evaluate the values and other relative data
        # of other's samples.
        replay_result = policy.compute_actions(
            other_batch[SampleBatch.CUR_OBS]
        )[2]

        other_batch[SampleBatch.VF_PREDS] = replay_result[SampleBatch.VF_PREDS]
        other_batch[BEHAVIOUR_LOGITS] = replay_result[BEHAVIOUR_LOGITS]

        other_batch[ACTION_LOGP], other_batch[ACTION_PROB] = \
            _compute_logp(
                other_batch[BEHAVIOUR_LOGITS],
                other_batch[SampleBatch.ACTIONS]
            )

        if policy.config[REPLAY_VALUES]:
            # replay values
            postprocess_diversity(policy, other_batch, others_batches)
            batches.append(
                postprocess_ppo_gae_replay(policy, other_batch, other_policy)
            )
        else:
            other_batch_raw = postprocess_diversity(
                policy, other_batch_raw, others_batches
            )
            batches.append(postprocess_ppo_gae(policy, other_batch_raw))

    for batch in batches:
        batch[Postprocessing.ADVANTAGES + "_unnormalized"] = batch[
            Postprocessing.ADVANTAGES].copy().astype(np.float32)
        if "debug_ratio" not in batch:
            assert "debug_fake_adv" not in batch
            batch['debug_fake_adv'] = batch['debug_ratio'] = np.zeros_like(
                batch['advantages'], dtype=np.float32
            )

    batch = SampleBatch.concat_samples(batches) if len(batches) != 1 \
        else batches[0]
    # episode.batch_builder.count = batch.count
    return batch


def _compute_advantages_for_diversity(
        rewards, last_r, gamma, lambda_, values, use_gae=True
):
    if use_gae:
        vpred_t = np.concatenate([values, np.array([last_r])])
        delta_t = (rewards + gamma * vpred_t[1:] - vpred_t[:-1])
        advantage = discount(delta_t, gamma * lambda_)
        value_target = (advantage + values).copy().astype(np.float32)
    else:
        rewards_plus_v = np.concatenate([rewards, np.array([last_r])])
        advantage = discount(rewards_plus_v, gamma)[:-1]
        value_target = np.zeros_like(advantage)
    advantage = advantage.copy().astype(np.float32)
    return advantage, value_target
