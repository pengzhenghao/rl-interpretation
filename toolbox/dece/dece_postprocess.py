from ray.rllib.agents.ppo.ppo_policy import postprocess_ppo_gae, ACTION_LOGP, \
    SampleBatch, BEHAVIOUR_LOGITS
from ray.rllib.evaluation.postprocessing import discount, Postprocessing

from toolbox.dece.utils import *

MY_LOGIT = "my_logits"
MY_LOGP = "my_logp"


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


def postprocess_vtrace(policy, sample_batch, others_batches, episode):
    """Add novelty and MY_LOGIT into batches."""
    batch = sample_batch.copy()
    if not policy.loss_initialized():
        batch[NOVELTY_REWARDS] = np.zeros_like(
            batch[SampleBatch.REWARDS], dtype=np.float32
        )
        batch[MY_LOGP] = np.zeros_like(batch[ACTION_LOGP], dtype=np.float32)
        batch[MY_LOGIT] = np.zeros_like(
            batch[BEHAVIOUR_LOGITS], dtype=np.float32
        )
        return batch

    batch[MY_LOGP] = batch[ACTION_LOGP].copy()
    batch[MY_LOGIT] = batch[BEHAVIOUR_LOGITS].copy()
    batch[NOVELTY_REWARDS] = policy.compute_novelty(
        batch, others_batches, use_my_logit=True
    )
    batches = [batch]
    for pid, (other_policy, other_batch_raw) in others_batches.items():
        if other_batch_raw is None:
            continue
        other_batch_raw = other_batch_raw.copy()

        # add action_logp
        replay_result = policy.compute_actions(
            other_batch_raw[SampleBatch.CUR_OBS]
        )[2]

        other_batch_raw[MY_LOGP], _ = \
            _compute_logp(
                replay_result[BEHAVIOUR_LOGITS],
                other_batch_raw[SampleBatch.ACTIONS]
            )

        other_batch_raw[MY_LOGIT] = \
            replay_result[BEHAVIOUR_LOGITS]

        other_batch_raw[NOVELTY_REWARDS] = policy.compute_novelty(
            other_batch_raw, others_batches, use_my_logit=True
        )

        batches.append(other_batch_raw)
    return SampleBatch.concat_samples(batches) if len(batches) != 1 \
        else batches[0]


def postprocess_no_replay_values(
        policy, sample_batch, others_batches, episode
):
    """Replay to collect logits. Pretend the samples from other agent is
    exactly my samples (no change values, advantages, value target)."""
    config = policy.config
    assert not config[REPLAY_VALUES]
    if not policy.loss_initialized():
        batch = postprocess_ppo_gae(policy, sample_batch)
        batch["abs_advantage"] = np.zeros_like(
            batch["advantages"], dtype=np.float32
        )
        batch['debug_ratio'] = np.zeros_like(
            batch["advantages"], dtype=np.float32
        )
        batch[NOVELTY_REWARDS] = np.zeros_like(
            batch["advantages"], dtype=np.float32
        )
        batch[NOVELTY_VALUE_TARGETS] = np.zeros_like(
            batch["advantages"], dtype=np.float32
        )
        batch[NOVELTY_ADVANTAGES] = np.zeros_like(
            batch["advantages"], dtype=np.float32
        )
        batch['other_action_logp'] = np.zeros_like(
            batch[ACTION_LOGP], dtype=np.float32
        )
        return batch

    batch = sample_batch.copy()
    batch = postprocess_ppo_gae(policy, batch)
    batch["abs_advantage"] = np.abs(batch[Postprocessing.ADVANTAGES])
    batch[MY_LOGIT] = batch[BEHAVIOUR_LOGITS]
    batch = postprocess_diversity(policy, batch, others_batches)
    batches = [batch]

    for pid, (other_policy, other_batch_raw) in others_batches.items():
        # The logic is that EVEN though we may use DISABLE or NO_REPLAY_VALUES,
        # but we still want to take a look of those statics.
        # Maybe in the future we can add knob to remove all such slowly stats.

        if other_batch_raw is None:
            continue

        other_batch_raw = other_batch_raw.copy()

        replay_result = policy.compute_actions(
            other_batch_raw[SampleBatch.CUR_OBS]
        )[2]

        other_batch_raw[MY_LOGIT] = replay_result[BEHAVIOUR_LOGITS]

        # the behaviour logits used for computing diversity is from other's
        # policy, and the comparing subject is other polies too
        # so there is a mess. We need to either (1) compute novelty use
        # my logit or (2) compute novelty with other's logit and compare to
        # my policy.
        # Maybe the first solution sound natural.
        other_batch_raw = postprocess_diversity(
            policy, other_batch_raw, others_batches
        )

        to_add_batch = postprocess_ppo_gae(policy, other_batch_raw)
        to_add_batch["abs_advantage"] = np.abs(
            to_add_batch[Postprocessing.ADVANTAGES]
        )
        batches.append(to_add_batch)

    for batch in batches:
        if ("debug_ratio" not in batch) and (not config[REPLAY_VALUES]):
            batch['debug_ratio'] = np.zeros_like(
                batch['advantages'], dtype=np.float32
            )
    batch = SampleBatch.concat_samples(batches) if len(batches) != 1 \
        else batches[0]
    return batch


def postprocess_dece(policy, sample_batch, others_batches, episode):
    # only TNB happen for all cases, so need to handle it for all cases.
    if policy.config[REPLAY_VALUES]:
        batch = postprocess_vtrace(
            policy, sample_batch, others_batches, episode
        )
    else:
        batch = postprocess_no_replay_values(
            policy, sample_batch, others_batches, episode
        )

    # if policy.loss_initialized():
    #     assert episode is not None
    #     logger.debug(
    #         "***** In postprocess, when loss is initialized, we modified the "
    #         "episode.batch_builder.count to the value of sampled contain "
    #         "in each policy's batch: the sum of the count of all possible "
    #         "batches. The count of the post-processed batch is {}, the sum "
    #         "of all possible batches is {}, the current count of batch "
    #         "builder is {}. This policy name is: {}.".format(
    #             batch.count, sum(b.count for _, b in others_batches.values(
    #             )) + sample_batch.count, episode.batch_builder.count,
    #             np.unique(sample_batch[SampleBatch.AGENT_INDEX])))
    #     print(
    #         "***** In postprocess, when loss is initialized, we modified the "
    #         "episode.batch_builder.count to the value of sampled contain "
    #         "in each policy's batch: the sum of the count of all possible "
    #         "batches. The count of the post-processed batch is {}, the sum "
    #         "of all possible batches is {}, the current count of batch "
    #         "builder is {}. The updated count of batch builder is {}. "
    #         "This policy name is: {}.".format(
    #             batch.count, sum(b.count for _, b in others_batches.values(
    #             )) + sample_batch.count, episode.batch_builder.count,
    #             episode.batch_builder.count + batch.count,
    #             np.unique(sample_batch[SampleBatch.AGENT_INDEX])))
    #     episode.batch_builder.count += batch.count
    del batch.data['new_obs']  # save memory
    del batch.data['action_prob']
    return batch


def postprocess_diversity(policy, batch, others_batches):
    completed = batch["dones"][-1]
    batch[NOVELTY_REWARDS] = policy.compute_novelty(
        batch, others_batches, use_my_logit=True
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
        value_target = np.zeros_like(advantage, dtype=np.float32)
    advantage = advantage.copy().astype(np.float32)
    return advantage, value_target
