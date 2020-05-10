import logging

import numpy as np
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution
from ray.rllib.utils import try_import_tf, try_import_tfp

logger = logging.getLogger(__name__)
tf = try_import_tf()
tfp = try_import_tfp()
tfd = tfp.distributions


class DeterministicMixture(TFActionDistribution):
    name = "DeterministicMixture"

    def __init__(self, inputs, model):
        self.k = model.model_config['custom_options']['num_components']
        input_length = inputs.shape.as_list()[1]
        action_length = int(input_length / self.k - 1)
        num_splits = [action_length * self.k, self.k]
        splits = tf.split(inputs, num_splits, axis=1)
        self.weight = splits[-1]
        self.mixture_dist = tfd.Categorical(
            logits=self.weight, allow_nan_stats=False
        )
        # self.means = tf.reshape(splits[0], [-1, self.k, action_length])
        self.means = tf.reshape(splits[0], [-1, action_length, self.k])
        self.components_dist = tfd.Deterministic(
            self.means,
            # validate_args=True,
            allow_nan_stats=False
        )
        self.mixture = tfd.MixtureSameFamily(
            mixture_distribution=self.mixture_dist,
            components_distribution=self.components_dist,
            # validate_args=True,
            allow_nan_stats=False
        )
        self.inputs = inputs
        self.action_length = action_length
        TFActionDistribution.__init__(self, inputs, model)

    def _build_sample_op(self):
        ret = self.mixture.sample()
        return ret

    def logp(self, x):
        return tf.zeros(tf.shape(self.weight)[0])

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        action_length = np.prod(action_space.shape)
        k = model_config["custom_options"]["num_components"]
        return action_length * k + k

    def deterministic_sample(self):
        # This is a workaround. The return is not really deterministic.
        return self.mixture.sample()

    def sampled_action_logp(self):
        return tf.zeros(tf.shape(self.weight)[0])


class GaussianMixture(TFActionDistribution):
    name = "GaussianMixture"

    def __init__(self, inputs, model):
        self.k = model.model_config['custom_options']['num_components']
        input_length = inputs.shape.as_list()[1]
        action_length = int((input_length / self.k - 1) / 2)
        num_splits = [action_length * self.k, action_length * self.k, self.k]

        splits = tf.split(inputs, num_splits, axis=1)
        log_stds = tf.reshape(splits[1], [-1, self.k, action_length])
        self.stds = tf.exp(log_stds)
        self.means = tf.reshape(splits[0], [-1, self.k, action_length])
        self.weight = splits[-1]

        self.mixture_dist = tfd.Categorical(
            logits=self.weight, allow_nan_stats=False
        )
        self.components_dist = tfd.MultivariateNormalDiag(
            self.means,  # One for each component.
            self.stds,
            allow_nan_stats=False
        )
        self.gaussian_mixture_model = tfd.MixtureSameFamily(
            mixture_distribution=self.mixture_dist,
            components_distribution=self.components_dist,
            allow_nan_stats=False
        )
        self.inputs = inputs
        TFActionDistribution.__init__(self, inputs, model)

    def logp(self, x):
        return self.gaussian_mixture_model.log_prob(x)

    # def entropy(self):
    #     return self.gaussian_mixture_model.entropy()

    def _build_sample_op(self):
        ret = self.gaussian_mixture_model.sample()
        return ret

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        action_length = np.prod(action_space.shape)
        k = model_config["custom_options"]["num_components"]
        return action_length * 2 * k + k

    def deterministic_sample(self):
        # This is a workaround. The return is not really deterministic.
        return self.gaussian_mixture_model.sample()


def register_mixture_action_distribution():
    ModelCatalog.register_custom_action_dist(
        GaussianMixture.name, GaussianMixture
    )
    ModelCatalog.register_custom_action_dist(
        DeterministicMixture.name, DeterministicMixture
    )
    print(
        "Successfully register GaussianMixture and DeterministicMixture "
        "action distribution."
    )


register_mixture_action_distribution()

if __name__ == '__main__':
    from ray import tune
    from toolbox import initialize_ray

    initialize_ray(test_mode=True, local_mode=True)

    tune.run(
        "TD3",
        # PPOTrainerWithoutKL,
        local_dir="/tmp/ray",
        name="DELETE_ME_TEST",
        config={
            "env": "BipedalWalker-v2",
            "log_level": "DEBUG",
            "model": {
                "custom_action_dist": GaussianMixture.name,
                "custom_options": {
                    "num_components": 7
                }
            }
        },
        stop={"timesteps_total": 10000}
    )
