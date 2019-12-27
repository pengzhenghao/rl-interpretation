import numpy as np

NOVELTY_REWARDS = "novelty_rewards"
NOVELTY_VALUES = "novelty_values"
NOVELTY_ADVANTAGES = "novelty_advantages"
NOVELTY_VALUE_TARGETS = "novelty_value_targets"


class RunningMean(object):
    """Implement the logic of Sun Hao running mean of rewards.
     Input is in batch form."""

    def __init__(self, num_policies):
        self.length = np.zeros((num_policies, 1))
        self.accumulated = np.zeros((num_policies, 1))
        self.num_policies = num_policies

    def __call__(self, x):
        x = np.asarray(x)
        assert x.ndim == 2
        assert x.shape[0] == self.num_policies
        # ([num_policies, batch size] + [num_policies, 1]) / (batch size + len)
        ret = (x + self.accumulated) / (x.shape[1] + self.length)
        self.accumulated += x.sum(axis=1)[:, np.newaxis]
        self.length += x.shape[1]
        return ret
