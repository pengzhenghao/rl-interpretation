from ray.rllib.agents.ppo.ppo_policy import postprocess_ppo_gae, ACTION_LOGP, \
    SampleBatch, BEHAVIOUR_LOGITS
from ray.rllib.evaluation.postprocessing import discount, Postprocessing

from toolbox.dece.utils import *

MY_LOGIT = "my_logits"
MY_LOGP = "my_logp"

# def postprocess_vtrace_diversity(policy, batch, others_batches,
# episode=None):
#     raise ValueError()
# completed = batch["dones"][-1]
# batch[NOVELTY_REWARDS] = policy.compute_novelty(
#     batch, others_batches, episode
# )
#
# if completed:
#     last_r_novelty = 0.0
# else:
#     next_state = []
#     for i in range(policy.num_state_tensors()):
#         next_state.append([batch["state_out_{}".format(i)][-1]])
#     last_r_novelty = policy._novelty_value(
#         batch[SampleBatch.NEXT_OBS][-1], batch[SampleBatch.ACTIONS][-1],
#         batch[NOVELTY_REWARDS][-1], *next_state
#     )
#
# # compute the advantages of novelty rewards
# batch = compute_vtrace(
#     batch,
#     last_r_novelty,
#     gamma=policy.config["gamma"],
#     clip_rho_threshold=policy.config['clip_rho_threshold'],
#     clip_pg_rho_threshold=policy.config['clip_pg_rho_threshold'],
#     use_diversity=True
# )
# return batch

# def postprocess_vtrace(policy, batch, other_policy=None):
#     completed = batch["dones"][-1]
#     if completed:
#         my_last_r = 0.0
#     else:
#         next_state = []
#         for i in range(policy.num_state_tensors()):
#             next_state.append([batch["state_out_{}".format(i)][-1]])
#         my_last_r = policy._value(
#             batch[SampleBatch.NEXT_OBS][-1], batch[SampleBatch.ACTIONS][-1],
#             batch[SampleBatch.REWARDS][-1], *next_state
#         )
#     batch = compute_vtrace(
#         batch,
#         my_last_r,
#         gamma=policy.config["gamma"],
#         clip_rho_threshold=policy.config['clip_rho_threshold'],
#         clip_pg_rho_threshold=policy.config['clip_pg_rho_threshold']
#     )
#     return batch

# def calculate_vtrace_minus(values, delta, ratio, gamma):
#     assert len(delta) > 0, (values, delta)
#     assert values.shape == delta.shape
#     y_n = np.zeros_like(delta)
#     y_n[-1] = delta[-1]
#     length = len(delta)
#     # the ratio mean c_i here.
#     # the delta mean v_{s+1} - V(x_{s+1})
#     for ind in range(length - 2, -1, -1):
#         # ind = 8, 7, 6, ..., 0 if length = 10
#         # y_n[ind] = delta[ind] + ratio[ind] * gamma * delta[ind + 1]
#         y_n[ind] = delta[ind] + ratio[ind] * gamma * y_n[ind + 1]
#     return y_n

# def compute_vtrace(
#         rollout,
#         my_last_r,
#         gamma=0.9,
#         clip_rho_threshold=1.0,
#         clip_pg_rho_threshold=1.0,
#         use_diversity=False
# ):
#     traj = {}
#     trajsize = len(rollout[SampleBatch.ACTIONS])
#     for key in rollout:
#         traj[key] = np.stack(rollout[key])
#
#     vf_preds = rollout[NOVELTY_VALUES] if use_diversity else rollout[
#         SampleBatch.VF_PREDS]
#     rewards = traj[NOVELTY_REWARDS] if use_diversity else traj[
#         SampleBatch.REWARDS]
#
#     assert SampleBatch.VF_PREDS in rollout, "Values not found!"
#     vpred_t = np.concatenate([vf_preds, np.array([my_last_r])])
#     ratio = np.exp(traj['action_logp'] - traj["other_action_logp"])
#     clipped_ratio = np.minimum(
#         ratio, clip_rho_threshold
#     # is \bar{pho} !
#     delta_t = clipped_ratio * (rewards + gamma * vpred_t[1:] - vpred_t[:-1])
#     cs = np.minimum(ratio, 1.0)
#     vs = calculate_vtrace_minus(vf_preds, delta_t, cs, gamma) + vf_preds
#     vs = np.concatenate([vs, np.array([my_last_r])])
#
#     clipped_pg_rhos = np.minimum(ratio, clip_pg_rho_threshold)
#     advantage = clipped_pg_rhos * (rewards + gamma * vs[1:] - vs[:-1])
#
#     if use_diversity:
#         traj[NOVELTY_ADVANTAGES] = advantage
#         traj["abs_novelty_advantage"] = np.abs(advantage)
#         # traj["debug_ratio"] = ratio
#         # traj["is_ratio"] = np.clip(
#         #     np.exp(traj["other_action_logp"] - traj["action_logp"]), 0,
#         2.0)
#         traj[NOVELTY_VALUE_TARGETS] = vs[:-1]
#         traj[NOVELTY_ADVANTAGES] = traj[NOVELTY_ADVANTAGES].copy().astype(
#             np.float32
#         )
#     else:
#         traj[Postprocessing.ADVANTAGES] = advantage
#         traj["abs_advantage"] = np.abs(advantage)
#         traj["debug_ratio"] = ratio
#         traj["is_ratio"] = np.clip(
#             np.exp(traj["action_logp"] - traj["other_action_logp"]), 0, 2.0
#         )
#         # np.exp(traj["other_action_logp"] - traj["action_logp"]), 0, 2.0)
#         traj[Postprocessing.VALUE_TARGETS] = vs[:-1]
#         traj[Postprocessing.ADVANTAGES
#              ] = traj[Postprocessing.ADVANTAGES].copy().astype(np.float32)
#
#     # assert all(val.shape[0] == trajsize for val in traj.values()), \
#     #     "Rollout stacked incorrectly!"
#     return SampleBatch(traj)

# def postprocess_ppo_gae_replay(policy, batch, other_policy):
#     """Adds the policy logits, VF preds, and advantages to the trajectory."""
#     completed = batch["dones"][-1]
#     if completed:
#         my_last_r = other_last_r = 0.0
#     else:
#         next_state = []
#         for i in range(policy.num_state_tensors()):
#             next_state.append([batch["state_out_{}".format(i)][-1]])
#         other_last_r = other_policy._value(
#             batch[SampleBatch.NEXT_OBS][-1], batch[SampleBatch.ACTIONS][-1],
#             batch[SampleBatch.REWARDS][-1], *next_state
#         )
#         my_last_r = policy._value(
#             batch[SampleBatch.NEXT_OBS][-1], batch[SampleBatch.ACTIONS][-1],
#             batch[SampleBatch.REWARDS][-1], *next_state
#         )
#     batch = compute_advantages_replay(
#         batch,
#         my_last_r,
#         other_last_r,
#         policy.config["gamma"],
#         policy.config["lambda"],
#         use_gae=policy.config["use_gae"]
#     )
#     return batch

# def compute_advantages_replay(
#         rollout, my_last_r, other_last_r, gamma=0.9, lambda_=1.0,
#         use_gae=True
# ):
#     """Given a rollout, compute its value targets and the advantage.
#
#     Args:
#         rollout (SampleBatch): SampleBatch of a single trajectory
#         last_r (float): Value estimation for last observation
#         gamma (float): Discount factor.
#         lambda_ (float): Parameter for GAE
#         use_gae (bool): Using Generalized Advantage Estimation
#
#     Returns:
#         SampleBatch (SampleBatch): Object with experience from rollout and
#             processed rewards.
#     """
#     traj = {}
#     trajsize = len(rollout[SampleBatch.ACTIONS])
#     for key in rollout:
#         traj[key] = np.stack(rollout[key])
#
#     if use_gae:
#         assert SampleBatch.VF_PREDS in rollout, "Values not found!"
#         vpred_t = np.concatenate(
#             [rollout[SampleBatch.VF_PREDS],
#              np.array([my_last_r])]
#         )
#         delta_t = traj[SampleBatch.REWARDS
#                        ] + gamma * vpred_t[1:] * (1 - lambda_) - vpred_t[:-1]
#         # other_vpred_t = np.concatenate(
#         #     [rollout["other_vf_preds"],
#         #      np.array([other_last_r])]
#         # )
#         # other_delta_t = (
#         #         traj[SampleBatch.REWARDS] + gamma * other_vpred_t[1:] -
#         #         other_vpred_t[:-1]
#         # )
#         # other_advantage = discount(other_delta_t, gamma * lambda_)
#
#         ratio = np.exp(traj['action_logp'] - traj["other_action_logp"])
#         # other_advantage = (other_advantage - other_advantage.mean()
#         #                    ) / max(1e-4, other_advantage.std())
#
#         # we put other's advantage in 'advantages' field. We need to make
#         sure
#         # this field is not used in future postprocessing.
#         # traj[Postprocessing.ADVANTAGES] = other_advantage
#
#         # other_delta_t = (
#         #     traj[SampleBatch.REWARDS] + gamma * other_vpred_t[1:] -
#         #     other_vpred_t[:-1]
#         # )
#         advantage = calculate_gae_advantage(
#             traj[SampleBatch.VF_PREDS], delta_t, ratio, lambda_, gamma
#         )
#
#         traj["abs_advantage"] = np.abs(advantage)
#         traj[Postprocessing.ADVANTAGES
#              ] = (advantage - advantage.mean()) / max(1e-4, advantage.std())
#         traj["debug_ratio"] = ratio
#
#         my_vpred_t = np.concatenate(
#             [rollout[SampleBatch.VF_PREDS],
#              np.array([my_last_r])]
#         )
#         assert ratio.shape == traj[SampleBatch.REWARDS].shape
#
#         clipped_ratio = np.clip(ratio, 0, 1.0)
#         value_target = (
#             clipped_ratio *
#             (traj[SampleBatch.REWARDS] + gamma * my_vpred_t[1:]) +
#             (1 - clipped_ratio) * (my_vpred_t[:-1])
#         )
#
#         traj[Postprocessing.VALUE_TARGETS] = value_target
#
#     else:
#         raise NotImplementedError()
#         # rewards_plus_v = np.concatenate(
#         #     [rollout[SampleBatch.REWARDS],
#         #      np.array([last_r])])
#         #
#         # ratio = np.exp(traj['action_logp'] - traj["other_action_logp"])
#         # assert_nan(ratio)
#         # delta = discount(rewards_plus_v, gamma)[:-1]
#         #
#         # assert delta.shape == ratio.shape
#         #
#         # traj[Postprocessing.ADVANTAGES] = ratio * delta
#         # traj[Postprocessing.VALUE_TARGETS] = np.zeros_like(
#         #     traj[Postprocessing.ADVANTAGES])
#
#     traj[Postprocessing.ADVANTAGES
#          ] = traj[Postprocessing.ADVANTAGES].copy().astype(np.float32)
#
#     assert all(val.shape[0] == trajsize for val in traj.values()), \
#         "Rollout stacked incorrectly!"
#     return SampleBatch(traj)

# def calculate_gae_advantage(values, delta, ratio, lambda_, gamma):
#     assert len(delta) > 0, (values, delta)
#     assert values.shape == delta.shape
#     y_n = np.zeros_like(delta)
#     y_n[-1] = delta[-1]
#     length = len(delta)
#     for ind in range(length - 2, -1, -1):
#         # ind = 8, 7, 6, ..., 0 if length = 10
#         y_n[ind] = delta[ind] + ratio[
#             ind + 1] * gamma * lambda_ * (y_n[ind + 1] + values[ind + 1])
#     return y_n


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


# def _clip_batch(other_batch, clip_action_prob_kl):
#     kl = get_kl_divergence(
#         source=other_batch[BEHAVIOUR_LOGITS],
#         target=other_batch["other_logits"],
#         mean=False
#     )
#
#     mask = kl < clip_action_prob_kl
#     length = len(mask)
#     info = {"kl": kl, "unclip_length": length, "length": len(mask)}
#
#     if not np.all(mask):
#         length = mask.argmin()
#         info['unclip_length'] = length
#         if length == 0:
#             return None, info
#         assert length < len(other_batch['action_logp'])
#         other_batch = other_batch.slice(0, length)
#
#     return other_batch, info


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


# def postprocess_replay_values(policy, sample_batch, others_batches, episode):
#     """Replay for logp, logits, values, value_target, advantage.
#     This is almost deprecated because we want to use vtrace"""
#     config = policy.config
#     if not policy.loss_initialized():
#         batch = postprocess_ppo_gae(policy, sample_batch)
#         batch["abs_advantage"] = np.zeros_like(
#             batch["advantages"], dtype=np.float32
#         )
#         batch['debug_ratio'] = np.zeros_like(
#             batch["advantages"], dtype=np.float32
#         )
#         batch[NOVELTY_REWARDS] = np.zeros_like(
#             batch["advantages"], dtype=np.float32
#         )
#         if policy.config[DIVERSITY_ENCOURAGING] and (not config[
#         REPLAY_VALUES]):
#             batch[NOVELTY_VALUE_TARGETS] = np.zeros_like(
#                 batch["advantages"], dtype=np.float32
#             )
#             batch[NOVELTY_ADVANTAGES] = np.zeros_like(
#                 batch["advantages"], dtype=np.float32
#             )
#             # batch['other_logits'] = np.zeros_like(
#             #     batch[BEHAVIOUR_LOGITS], dtype=np.float32
#             # )
#             batch['other_action_logp'] = np.zeros_like(
#                 batch[ACTION_LOGP], dtype=np.float32
#             )
#         return batch
#
#     batch = sample_batch.copy()
#     # batch["other_logits"] = batch[BEHAVIOUR_LOGITS].copy()
#     batch["other_action_logp"] = batch[ACTION_LOGP].copy()
#     tmp_batch = postprocess_ppo_gae(policy, batch)
#     value = tmp_batch[Postprocessing.ADVANTAGES]
#     standardized = (value - value.mean()) / max(1e-4, value.std())
#     tmp_batch[Postprocessing.ADVANTAGES] = standardized
#     tmp_batch = postprocess_diversity(
#         policy, tmp_batch, others_batches, use_my_logit=False
#     )
#     tmp_batch["abs_advantage"] = np.abs(tmp_batch[Postprocessing.ADVANTAGES])
#     batches = [tmp_batch]
#
#     if config[ONLY_TNB]:
#         if ("debug_ratio" not in batch) and (not config[REPLAY_VALUES]):
#             # assert "debug_fake_adv" not in batch
#             batch['debug_ratio'] = np.zeros_like(
#                 batch['advantages'], dtype=np.float32
#             )
#         return batch
#
#     for pid, (other_policy, other_batch_raw) in others_batches.items():
#         # The logic is that EVEN though we may use DISABLE or
#         NO_REPLAY_VALUES,
#         # but we still want to take a look of those statics.
#         # Maybe in the future we can add knob to remove all such slowly
#         stats.
#
#         if other_batch_raw is None:
#             continue
#
#         other_batch_raw = other_batch_raw.copy()
#         other_batch = other_batch_raw.copy()
#
#         # four fields that we will overwrite.
#         # Two additional: advantages / value target
#         other_batch["other_action_logp"] = other_batch[ACTION_LOGP].copy()
#         other_batch["other_action_prob"] = other_batch[ACTION_PROB].copy()
#         # other_batch["other_logits"] = other_batch[BEHAVIOUR_LOGITS].copy()
#         other_batch["other_vf_preds"] = other_batch[SampleBatch.VF_PREDS
#                                                     ].copy()
#
#         # use my policy to evaluate the values and other relative data
#         # of other's samples.
#         replay_result = policy.compute_actions(
#             other_batch[SampleBatch.CUR_OBS]
#         )[2]
#
#         other_batch[SampleBatch.VF_PREDS] = replay_result[
#         SampleBatch.VF_PREDS]
#         other_batch[BEHAVIOUR_LOGITS] = replay_result[BEHAVIOUR_LOGITS]
#
#         if policy.config[USE_DIVERSITY_VALUE_NETWORK]:
#             other_batch[NOVELTY_VALUES] = replay_result[NOVELTY_VALUES]
#
#         other_batch[ACTION_LOGP], other_batch[ACTION_PROB] = \
#             _compute_logp(
#                 other_batch[BEHAVIOUR_LOGITS],
#                 other_batch[SampleBatch.ACTIONS]
#             )
#         other_batch = postprocess_diversity(
#             policy, other_batch, others_batches
#         )
#         to_add_batch = postprocess_ppo_gae_replay(
#             policy, other_batch, other_policy
#         )
#         batches.append(to_add_batch)
#
#     for batch in batches:
#         # batch[Postprocessing.ADVANTAGES + "_unnormalized"] = batch[
#         #     Postprocessing.ADVANTAGES].copy().astype(np.float32)
#         if ("debug_ratio" not in batch) and (not config[REPLAY_VALUES]):
#             # assert "debug_fake_adv" not in batch
#             batch['debug_ratio'] = np.zeros_like(
#                 batch['advantages'], dtype=np.float32
#             )
#
#     return SampleBatch.concat_samples(batches) if len(batches) != 1 \
#         else batches[0]


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

    # if config[ONLY_TNB]:
    #     if ("debug_ratio" not in batch) and (not config[REPLAY_VALUES]):
    #         # assert "debug_fake_adv" not in batch
    #         batch['debug_ratio'] = np.zeros_like(
    #             batch['advantages'], dtype=np.float32
    #         )
    #     return batch

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


def postprocess_dece(policy, *args, **kwargs):
    # only TNB happen for all cases, so need to handle it for all cases.
    if policy.config[REPLAY_VALUES]:
        batch = postprocess_vtrace(policy, *args, **kwargs)
    # elif policy.config[REPLAY_VALUES]:
    #     batch = postprocess_replay_values(policy, *args, **kwargs)
    else:
        batch = postprocess_no_replay_values(policy, *args, **kwargs)
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
