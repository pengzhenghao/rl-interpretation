import numpy as np
from ray.rllib.evaluation.postprocessing import discount
from ray.rllib.policy.sample_batch import SampleBatch


class Postprocessing(object):
    """Constant definitions for postprocessing."""
    ADVANTAGES = "advantages"
    VALUE_TARGETS = "value_targets"
    OTHER_ACTION_LOGP = "other_action_logp"
    OTHER_ACTION_PROB = "other_action_prob"


def postprocess_ppo_gae_replay(policy, sample_batch, other_policy):
    """Adds the policy logits, VF preds, and advantages to the trajectory."""
    completed = sample_batch["dones"][-1]
    if completed:
        my_last_r = other_last_r = 0.0
    else:
        next_state = []
        for i in range(policy.num_state_tensors()):
            next_state.append([sample_batch["state_out_{}".format(i)][-1]])
        other_last_r = other_policy._value(
            sample_batch[SampleBatch.NEXT_OBS][-1],
            sample_batch[SampleBatch.ACTIONS][-1],
            sample_batch[SampleBatch.REWARDS][-1], *next_state
        )
        my_last_r = policy._value(
            sample_batch[SampleBatch.NEXT_OBS][-1],
            sample_batch[SampleBatch.ACTIONS][-1],
            sample_batch[SampleBatch.REWARDS][-1], *next_state
        )
    batch = compute_advantages_replay(
        sample_batch,
        my_last_r,
        other_last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"],
        clip_action_prob_ratio=policy.config["clip_action_prob_ratio"]
    )
    return batch


def compute_advantages_replay(
        rollout,
        my_last_r,
        other_last_r,
        gamma=0.9,
        lambda_=1.0,
        use_gae=True,
        clip_action_prob_ratio=1
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
        other_delta_t = (traj[SampleBatch.REWARDS] + gamma * other_vpred_t[1:]
                         - other_vpred_t[:-1])
        other_advantage = discount(other_delta_t, gamma * lambda_)
        other_advantage = (other_advantage - other_advantage.mean()) / max(
            1e-4, other_advantage.std())
        # we put other's advantage in 'advantages' field. We need to make sure
        # this field is not used in future postprocessing.
        traj[Postprocessing.ADVANTAGES] = other_advantage

        # This formula for the advantage comes
        # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
        # traj[Postprocessing.ADVANTAGES] = discount(delta_t, gamma * lambda_)
        # advantage = ratio * delta_t
        # On the contrary, the use the naive form (single Bellman equation) to
        # compute the value target
        # advantage = calculate_gae_advantage(
        #     traj[SampleBatch.VF_PREDS], delta_t, ratio, lambda_, gamma
        # )

        # Ratio is almost deprecated. We only use it to compute value target.
        ratio = np.exp(traj['action_logp'] - traj["other_action_logp"])
        ratio = np.clip(ratio, 0.0, clip_action_prob_ratio)
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
        value_target = \
            ratio * (traj[SampleBatch.REWARDS] + gamma * my_vpred_t[1:])
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
