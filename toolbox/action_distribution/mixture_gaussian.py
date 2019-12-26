import logging

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution

logger = logging.getLogger(__name__)
tfd = tfp.distributions
"""This codes is still contains bugs. But I can not find it.

The most significant symptom is the NaN may raise as the sample, which
cause BipedalWalker-v2 raise "AssertionError: r.LengthSquared() > 0.0f"
error and Mujoco env raise "mujoco_py.builder.MujocoException: Unknown
 warning type Time = 0.0000.Check for NaN in simulation." error.

But I have not really try to look at does it really yield NaN as sample.

Anyway, this branch is closed. 
"""


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


def register_gaussian_mixture():
    ModelCatalog.register_custom_action_dist(
        GaussianMixture.name, GaussianMixture
    )


register_gaussian_mixture()

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