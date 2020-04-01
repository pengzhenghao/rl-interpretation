import logging

import numpy as np
import ray
from ray.rllib.agents.es import utils
from ray.rllib.agents.ppo.ppo import warn_about_bad_reward_scales

logger = logging.getLogger(__name__)


# Optimizer is copied from ray.rllib.agents.es.optimizers
class Optimizer:
    def __init__(self, num_parameters):
        # self.policy = policy
        # self.dim = policy._variables.get_flat().size
        self.dim = num_parameters
        self.t = 0

    def update(self, globalg, theta):
        self.t += 1
        step = self._compute_step(globalg)
        # theta = self.policy._variables.get_flat()
        ratio = np.linalg.norm(step) / np.linalg.norm(theta)
        return theta + step, ratio

    def _compute_step(self, globalg):
        raise NotImplementedError


class Adam(Optimizer):
    def __init__(self, policy, stepsize=0.01, beta1=0.9, beta2=0.999,
                 epsilon=1e-08):
        Optimizer.__init__(self, policy)
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)

    def _compute_step(self, globalg):
        a = self.stepsize * (np.sqrt(1 - self.beta2 ** self.t) /
                             (1 - self.beta1 ** self.t))
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step


def get_flat(policy):
    return policy._variables.get_flat()


def run_evolution_strategies(trainer, result):
    warn_about_bad_reward_scales(trainer, result)  # original function

    if not hasattr(trainer, "update_policy_counter"):
        # This should be the first iter?
        trainer.update_policy_counter = 1
        trainer._last_update_weights = get_flat(trainer.get_policy("agent0"))

        trainer.debug_last_update_weights = trainer.get_weights("agent0")[
            "agent0"]

        trainer._es_optimizer = Adam(trainer._last_update_weights.size)

    rewards = result['policy_reward_mean']
    steps = result['info']['num_steps_trained']

    for rk, r in rewards.items():
        assert np.isscalar(r), \
            "Invalid reward happen! Should we skip this update?"

    update_steps = trainer.config['update_steps']
    if update_steps == "baseline":
        # Never enter the ES synchronization if set update_steps to baseline.
        update_steps = float('+inf')
    else:
        assert isinstance(update_steps, int)

    if steps > update_steps * trainer.update_policy_counter:
        best_agent = max(rewards, key=lambda x: rewards[x])
        returns = np.array(list(rewards.values()))
        proc_noisy_returns = utils.compute_centered_ranks(returns)
        weights_diff = {}
        for pid, p in trainer.workers.local_worker().policy_map.items():
            weights_diff[pid] = get_flat(p) - trainer._last_update_weights

        # Compute and take a step.
        g, count = utils.batched_weighted_sum(
            proc_noisy_returns,
            (weights_diff[pid] for pid in rewards.keys()),
            batch_size=500)  # batch_size 500 always greater # of policy 10
        g /= returns.size

        # Compute the new weights theta.
        theta = trainer._last_update_weights  # Old weights
        new_theta, update_ratio = trainer._es_optimizer.update(
            -g + 0.005 * theta, theta)
        theta_id = ray.put(new_theta)

        def _spawn_policy(policy, policy_id):
            new_weights = ray.get(theta_id)
            policy._variables.set_flat(new_weights)
            print("Sync {}.".format(policy_id))

        # set to policies on local worker. Then all polices would be the same.
        trainer.workers.local_worker().foreach_policy(_spawn_policy)

        info = {
            "weights_norm": np.square(theta).sum(),
            "grad_norm": np.square(g).sum(),
            "update_ratio": update_ratio,
            "update_policy_counter": trainer.update_policy_counter
        }
        result["evolution_info"] = info

        msg = "Current num_steps_trained is {}, exceed last update steps {}" \
              " (our update interval is {}). Current best agent is <{}> " \
              "with reward {:.4f}. We spawn it to others: {}.".format(
            steps, trainer.update_policy_counter * update_steps, update_steps,
            best_agent, rewards[best_agent], rewards
        )
        print(msg)
        logger.info(msg)
        trainer._last_update_weights = new_theta.copy()
        trainer.update_policy_counter += 1

    result['update_policy_counter'] = trainer.update_policy_counter
    result['update_policy_threshold'] = trainer.update_policy_counter * \
                                        update_steps
