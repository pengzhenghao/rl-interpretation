import numpy as np
import scipy.signal
from ray.rllib.policy.sample_batch import SampleBatch


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class Postprocessing(object):
    """Constant definitions for postprocessing."""

    ADVANTAGES = "advantages"
    VALUE_TARGETS = "value_targets"
    OTHER_ACTION_LOGP = "other_action_logp"
    OTHER_ACTION_PROB = "other_action_prob"


def assert_nan(arr):
    assert not np.any(np.isnan(arr.astype(np.float32))), arr


def postprocess_ppo_gae_replay(policy,
                               sample_batch):
    """Adds the policy logits, VF preds, and advantages to the trajectory."""

    assert Postprocessing.OTHER_ACTION_PROB in sample_batch
    assert Postprocessing.OTHER_ACTION_LOGP in sample_batch

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
    batch = compute_advantages_replay(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"])
    return batch


def compute_advantages_replay(rollout, last_r, gamma=0.9, lambda_=1.0,
                              use_gae=True):
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
        vpred_t = np.concatenate(
            [rollout[SampleBatch.VF_PREDS],
             np.array([last_r])])
        delta_t = (
                traj[SampleBatch.REWARDS] + gamma * vpred_t[1:] - vpred_t[:-1])
        # This formula for the advantage comes
        # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438

        # traj[Postprocessing.ADVANTAGES] = discount(delta_t, gamma * lambda_)

        ratio = np.exp(traj['action_logp'] - traj["other_action_logp"])

        assert_nan(ratio)

        advantage = calculate_gae_advantage(delta_t, ratio, lambda_, gamma)

        assert_nan(advantage)

        traj[Postprocessing.ADVANTAGES] = advantage

        value_target = (
                traj[Postprocessing.ADVANTAGES] +
                traj[SampleBatch.VF_PREDS]).copy().astype(np.float32)

        traj[Postprocessing.VALUE_TARGETS] = value_target
    else:
        raise NotImplementedError()
        # rewards_plus_v = np.concatenate(
        #     [rollout[SampleBatch.REWARDS],
        #      np.array([last_r])])
        # traj[Postprocessing.ADVANTAGES] = discount(rewards_plus_v, gamma)[
        # :-1]
        # traj[Postprocessing.VALUE_TARGETS] = np.zeros_like(
        #     traj[Postprocessing.ADVANTAGES])

    traj[Postprocessing.ADVANTAGES] = traj[
        Postprocessing.ADVANTAGES].copy().astype(np.float32)

    assert all(val.shape[0] == trajsize for val in traj.values()), \
        "Rollout stacked incorrectly!"
    return SampleBatch(traj)


def calculate_gae_advantage(delta, ratio, lambda_, gamma):
    y_n = np.empty_like(delta)
    y_n[-1] = ratio[-1] * delta[-1]
    length = len(delta)
    for ind in range(length - 2, -1, -1):
        # ind = 8, 7, 6, ..., 0 if length = 10
        y_n[ind] = ratio[ind] * (delta[ind] + gamma * lambda_ * y_n[ind + 1])
    return y_n


def test_calculate_gae_advantage(n=1000):
    length = 100
    for i in range(n):
        delta = np.random.random((length,))  # suppose it's old advantage
        ratio = np.random.random((length,))  # suppose it's the ratio of prob
        lambda_ = np.random.random()
        gamma = np.random.random()
        adv = calculate_gae_advantage(delta, ratio, lambda_, gamma)
        cor = _calculate_gae_advantage_correct(delta, ratio, lambda_, gamma)
        np.testing.assert_almost_equal(adv, cor)


def _calculate_gae_advantage_correct(delta, ratio, lambda_, gamma):
    length = len(delta)
    prev = delta[-1] * ratio[-1]
    correct = [prev]
    for i in range(1, length):
        ind = length - i - 1
        prev = (prev * gamma * lambda_ + delta[ind]) * ratio[ind]
        correct.insert(0, prev)
    return np.array(correct)
