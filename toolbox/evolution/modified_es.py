import logging
import time

import numpy as np
import ray
from ray.rllib.agents import with_common_config
from ray.rllib.agents.es import optimizers, policies, utils
from ray.rllib.agents.es.es import ESTrainer, create_shared_noise, Result
from ray.rllib.agents.es.policies import get_filter, tf, ModelCatalog, \
    GenericPolicy
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.policy.tf_policy import SampleBatch

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = with_common_config(
    {
        "l2_coeff": 0.005,
        "noise_stdev": 0.02,
        "episodes_per_batch": 1000,
        "train_batch_size": 10000,
        "eval_prob": 0.003,
        "return_proc_mode": "centered_rank",
        "num_workers": 10,
        "stepsize": 0.01,
        "observation_filter": "NoFilter",
        "noise_size": 250000000,
        "report_length": 10,
        "optimizer_type": "adam"  # must in [adam, sgd]
    }
)


class SharedNoiseTable(object):
    def __init__(self, noise):
        if isinstance(noise, ray.local_mode_manager.LocalModeObjectID):
            self.noise = noise.value
        else:
            self.noise = noise
        assert self.noise.dtype == np.float32

    def get(self, i, dim):
        return self.noise[i:i + dim]

    def sample_index(self, dim):
        return np.random.randint(0, len(self.noise) - dim + 1)

    def get_delta(self, dim):
        idx = self.sample_index(dim)
        return idx, self.get(idx, dim)


@ray.remote
class Worker:
    def __init__(
            self, config, policy_params, env_creator, noise,
            min_task_runtime=0.2
    ):
        self.min_task_runtime = min_task_runtime
        self.config = config
        self.policy_params = policy_params
        self.noise = SharedNoiseTable(noise)

        self.env = env_creator(config["env_config"])
        from ray.rllib import models
        self.preprocessor = models.ModelCatalog.get_preprocessor(
            self.env, config["model"]
        )

        self.sess = utils.make_session(single_threaded=True)

        with tf.name_scope(DEFAULT_POLICY_ID):
            self.policy = GenericGaussianPolicy(
                self.sess, self.env.action_space, self.env.observation_space,
                self.preprocessor, config["observation_filter"],
                config["model"], **policy_params
            )

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

    def rollout(self, timestep_limit, add_noise=True):
        rollout_rewards, rollout_length = policies.rollout(
            self.policy,
            self.env,
            timestep_limit=timestep_limit,
            add_noise=add_noise
        )
        return rollout_rewards, rollout_length

    def do_rollouts(self, params, timestep_limit=None):
        # Set the network weights.
        self.policy.set_weights(params)

        noise_indices, returns, sign_returns, lengths = [], [], [], []
        eval_returns, eval_lengths = [], []

        # Perform some rollouts with noise.
        task_tstart = time.time()
        while (len(noise_indices) == 0
               or time.time() - task_tstart < self.min_task_runtime):

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
                    noise_index, self.policy.num_params
                )

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
                     np.sign(rewards_neg).sum()]
                )
                lengths.append([lengths_pos, lengths_neg])

        return Result(
            noise_indices=noise_indices,
            noisy_returns=returns,
            sign_noisy_returns=sign_returns,
            noisy_lengths=lengths,
            eval_returns=eval_returns,
            eval_lengths=eval_lengths
        )


class GenericGaussianPolicy(GenericPolicy):
    def __init__(
            self,
            sess,
            action_space,
            obs_space,
            preprocessor,
            observation_filter,
            model_options,
            action_noise_std=0.0
    ):
        self.sess = sess
        self.action_space = action_space
        self.action_noise_std = action_noise_std
        self.preprocessor = preprocessor
        self.observation_filter = get_filter(
            observation_filter, self.preprocessor.shape
        )
        self.inputs = tf.placeholder(
            tf.float32, [None] + list(self.preprocessor.shape)
        )

        # dist_class, dist_dim = ModelCatalog.get_action_dist(
        #     self.action_space, model_options, dist_type="deterministic")
        # dist_class, dist_dim = ModelCatalog.get_action_dist(
        #     self.action_space, model_options
        # )

        model_config = model_options
        self._dist_class, logit_dim = ModelCatalog.get_action_dist(
            action_space, model_config
        )

        with tf.name_scope(DEFAULT_POLICY_ID):
            self.model = ModelCatalog.get_model_v2(
                obs_space,
                action_space,
                logit_dim,
                model_config,
                framework="tf"
            )
        model_out, self._state_out = self.model(
            {SampleBatch.CUR_OBS: self.inputs}
        )
        dist = self._dist_class(model_out, self.model)
        self.model_out = model_out
        self.sampler = dist.sample()
        self.variables = ray.experimental.tf_utils.TensorFlowVariables(
            [], self.sess, self.model.variables()
        )
        self.num_params = sum(
            np.prod(variable.shape.as_list())
            for _, variable in self.variables.variables.items()
        )
        self.sess.run(tf.global_variables_initializer())

    def compute_actions(self, observation):
        observation = self.preprocessor.transform(observation)
        observation = self.observation_filter(observation[None], update=False)
        logit = self.sess.run(
            self.model_out, feed_dict={self.inputs: observation}
        )
        return logit

    def set_weights(self, x):
        if isinstance(x, dict):
            self.variables.set_weights(x)
        elif isinstance(x, np.ndarray):
            self.variables.set_flat(x)
        else:
            raise ValueError(
                "Wrong type when setting weights in policy: ", type(x)
            )

    def get_weights(self):
        # return self.variables.get_weights()
        return self.variables.get_flat()


class GaussianESTrainer(ESTrainer):
    _default_config = DEFAULT_CONFIG
    _name = "GaussianES"

    @staticmethod
    def with_updates(after_init):
        class return_cls(GaussianESTrainer):
            def __init__(self, *args, **kwargs):
                super(return_cls, self).__init__(*args, **kwargs)
                after_init(self)

        return return_cls

    def get_weights(self, policies=None):
        return {DEFAULT_POLICY_ID: self.policy.get_weights()}

    def set_weights(self, weights):
        assert len(weights) == 1
        self.policy.set_weights(weights[DEFAULT_POLICY_ID])

    def get_policy(self, _=None):
        return self.policy

    def compute_action(
            self,
            observation,
            state=None,
            prev_action=None,
            prev_reward=None,
            info=None,
            policy_id=DEFAULT_POLICY_ID,
            full_fetch=False
    ):
        return super().compute_action(observation)

    def _init(self, config, env_creator):
        policy_params = {"action_noise_std": 0.01}

        env = env_creator(config["env_config"])
        from ray.rllib import models
        preprocessor = models.ModelCatalog.get_preprocessor(env)

        self.sess = utils.make_session(single_threaded=False)
        self.policy = GenericGaussianPolicy(
            self.sess, env.action_space, env.observation_space, preprocessor,
            config["observation_filter"], config["model"], **policy_params
        )
        if config["optimizer_type"] == "adam":
            self.optimizer = optimizers.Adam(self.policy, config["stepsize"])
        elif config["optimizer_type"] == "sgd":
            self.optimizer = optimizers.SGD(self.policy, config["stepsize"])
        else:
            raise ValueError("optimizer must in [adam, sgd].")
        self.report_length = config["report_length"]

        # Create the shared noise table.
        logger.info("Creating shared noise table.")
        noise_id = create_shared_noise.remote(config["noise_size"])
        self.noise = SharedNoiseTable(ray.get(noise_id))

        # Create the actors.
        logger.info("Creating actors.")
        self._workers = [
            Worker.remote(config, policy_params, env_creator, noise_id)
            for _ in range(config["num_workers"])
        ]

        self.episodes_so_far = 0
        self.reward_list = []
        self.tstart = time.time()


if __name__ == '__main__':
    from toolbox import initialize_ray

    initialize_ray(test_mode=True, local_mode=True)
    config = {
        "num_workers": 3,
        "episodes_per_batch": 5,
        "train_batch_size": 150,
        "observation_filter": "NoFilter",
        "noise_size": 1000000
    }
    agent = GaussianESTrainer(config, "BipedalWalker-v2")

    agent.train()
