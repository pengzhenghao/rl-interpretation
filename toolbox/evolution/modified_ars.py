import logging
import time

import numpy as np
import ray
from ray.rllib.agents import with_common_config
from ray.rllib.agents.ars import optimizers, policies, utils
from ray.rllib.agents.ars.ars import create_shared_noise, ARSTrainer, Result
from ray.rllib.agents.ars.policies import tf
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID

from toolbox.evolution.modified_es import GenericGaussianPolicy, \
    SharedNoiseTable

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = with_common_config({
    "noise_stdev": 0.02,  # std deviation of parameter noise
    "num_rollouts": 32,  # number of perturbs to try
    "rollouts_used": 32,  # number of perturbs to keep in gradient estimate
    "num_workers": 2,
    "sgd_stepsize": 0.01,  # sgd step-size
    "observation_filter": "NoFilter",
    "noise_size": 250000000,
    "eval_prob": 0.03,  # probability of evaluating the parameter rewards
    "report_length": 10,  # how many of the last rewards we average over
    "offset": 0,
})


@ray.remote
class Worker:
    def __init__(self,
                 config,
                 env_creator,
                 noise,
                 min_task_runtime=0.2):
        self.min_task_runtime = min_task_runtime
        self.config = config
        self.noise = SharedNoiseTable(noise)

        self.env = env_creator(config["env_config"])
        from ray.rllib import models
        self.preprocessor = models.ModelCatalog.get_preprocessor(self.env)

        self.sess = utils.make_session(single_threaded=True)

        with tf.name_scope(DEFAULT_POLICY_ID):
            self.policy = GenericGaussianPolicy(
                self.sess, self.env.action_space, self.env.observation_space,
                self.preprocessor, config["observation_filter"],
                config["model"])

    @property
    def filters(self):
        return {DEFAULT_POLICY_ID: self.policy.get_filter()}

    def sync_filters(self, new_filters):
        for k in self.filters:
            self.filters[k].sync(new_filters[k])

    def get_filters(self, flush_after=False):
        return_filters = {}
        for k, f in self.filters.items():
            return_filters[k] = f.as_serializable()
            if flush_after:
                f.clear_buffer()
        return return_filters

    def rollout(self, timestep_limit, add_noise=False):
        rollout_rewards, rollout_length = policies.rollout(
            self.policy,
            self.env,
            timestep_limit=timestep_limit,
            add_noise=add_noise,
            offset=self.config["offset"])
        return rollout_rewards, rollout_length

    def do_rollouts(self, params, timestep_limit=None):
        # Set the network weights.
        self.policy.set_weights(params)

        noise_indices, returns, sign_returns, lengths = [], [], [], []
        eval_returns, eval_lengths = [], []

        # Perform some rollouts with noise.
        while (len(noise_indices) == 0):
            if np.random.uniform() < self.config["eval_prob"]:
                # Do an evaluation run with no perturbation.
                self.policy.set_weights(params)
                rewards, length = self.rollout(timestep_limit, add_noise=False)
                eval_returns.append(rewards.sum())
                eval_lengths.append(length)
            else:
                # Do a regular run with parameter perturbations.
                noise_index = self.noise.sample_index(self.policy.num_params)

                perturbation = self.config["noise_stdev"] * self.noise.get(
                    noise_index, self.policy.num_params)

                # These two sampling steps could be done in parallel on
                # different actors letting us update twice as frequently.
                self.policy.set_weights(params + perturbation)
                rewards_pos, lengths_pos = self.rollout(timestep_limit)

                self.policy.set_weights(params - perturbation)
                rewards_neg, lengths_neg = self.rollout(timestep_limit)

                noise_indices.append(noise_index)
                returns.append([rewards_pos.sum(), rewards_neg.sum()])
                sign_returns.append(
                    [np.sign(rewards_pos).sum(),
                     np.sign(rewards_neg).sum()])
                lengths.append([lengths_pos, lengths_neg])

        return Result(
            noise_indices=noise_indices,
            noisy_returns=returns,
            sign_noisy_returns=sign_returns,
            noisy_lengths=lengths,
            eval_returns=eval_returns,
            eval_lengths=eval_lengths)


class GaussianARSTrainer(ARSTrainer):
    _default_config = DEFAULT_CONFIG
    _name = "GaussianARS"

    def get_weights(self, policies=None):
        return {DEFAULT_POLICY_ID: self.policy.get_weights()}

    def set_weights(self, weights):
        assert len(weights) == 1
        self.policy.set_weights(weights[DEFAULT_POLICY_ID])

    def get_policy(self, _=None):
        return self.policy

    def compute_action(self, observation, state=None, prev_action=None,
                       prev_reward=None, info=None,
                       policy_id=DEFAULT_POLICY_ID, full_fetch=False):
        return super().compute_action(observation)

    def _init(self, config, env_creator):
        # PyTorch check.
        if config["use_pytorch"]:
            raise ValueError(
                "ARS does not support PyTorch yet! Use tf instead."
            )

        env = env_creator(config["env_config"])
        from ray.rllib import models
        preprocessor = models.ModelCatalog.get_preprocessor(env)

        self.sess = utils.make_session(single_threaded=False)
        self.policy = GenericGaussianPolicy(
            self.sess, env.action_space, env.observation_space, preprocessor,
            config["observation_filter"], config["model"])

        self.optimizer = optimizers.SGD(self.policy, config["sgd_stepsize"])

        self.rollouts_used = config["rollouts_used"]
        self.num_rollouts = config["num_rollouts"]
        self.report_length = config["report_length"]

        # Create the shared noise table.
        logger.info("Creating shared noise table.")
        noise_id = create_shared_noise.remote(config["noise_size"])
        self.noise = SharedNoiseTable(ray.get(noise_id))

        # Create the actors.
        logger.info("Creating actors.")
        self.workers = [
            Worker.remote(config, env_creator, noise_id)
            for _ in range(config["num_workers"])
        ]

        self.episodes_so_far = 0
        self.reward_list = []
        self.tstart = time.time()


if __name__ == '__main__':
    from toolbox import initialize_ray

    initialize_ray(test_mode=True, local_mode=True)
    config = {"num_workers": 3, "train_batch_size": 150,
              "observation_filter": "NoFilter",
              "noise_size": 1000000}
    agent = GaussianARSTrainer(config, "BipedalWalker-v2")

    agent.train()
