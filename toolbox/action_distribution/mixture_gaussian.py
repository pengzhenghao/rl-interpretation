import numpy as np
import tensorflow as tf
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution


class GaussianMixture(TFActionDistribution):
    """Action distribution where each vector element is a gaussian.

    The first half of the input vector defines the gaussian means, and the
    second half the gaussian standard deviations.
    """

    name = "GaussianMixture"

    def __init__(self, inputs, model):
        self.k = model.model_config['custom_options']['num_components']
        num_splits = inputs.shape.as_list()[1] / self.k
        splits = tf.split(inputs, num_splits, axis=1)

        self.means = splits[:-1:2]
        self.log_stds = splits[1::2]
        self.weight = splits[-1]

        self.std = [tf.exp(log_std) for log_std in self.log_stds]

        # self.mean = mean
        # self.log_std = log_std
        # self.std = tf.exp(log_std)
        TFActionDistribution.__init__(self, inputs, model)

    def logp(self, x):
        raise NotImplementedError()
        # return (-0.5 * tf.reduce_sum(
        #     tf.square((x - self.mean) / self.std), reduction_indices=[1]) -
        #         0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[1]) -
        #         tf.reduce_sum(self.log_std, reduction_indices=[1]))

    def kl(self, other):
        raise NotImplementedError()
        # return tf.reduce_sum(
        #     other.log_std - self.log_std +
        #     (tf.square(self.std) + tf.square(self.mean - other.mean)) /
        #     (2.0 * tf.square(other.std)) - 0.5,
        #     reduction_indices=[1])

    def entropy(self):
        raise NotImplementedError()
        # return tf.reduce_sum(
        #     .5 * self.log_std + .5 * np.log(2.0 * np.pi * np.e),
        #     reduction_indices=[1])

    def _build_sample_op(self):
        raise NotImplementedError()
        # return self.mean + self.std * tf.random_normal(tf.shape(self.mean))

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        action_length = np.prod(action_space.shape)
        k = model_config["custom_options"]["num_components"]
        return action_length * 2 * k + k


def register_gaussian_mixture():
    ModelCatalog.register_custom_action_dist(
        GaussianMixture.name, GaussianMixture)


register_gaussian_mixture()

if __name__ == '__main__':
    from ray import tune
    from toolbox import initialize_ray

    initialize_ray(test_mode=True, local_mode=True)

    tune.run(
        "PPO",
        local_dir="/tmp/ray",
        name="DELETE_ME_TEST",
        config={
            "env": "CartPole-v0",
            "model": {
                "custom_action_dist": GaussianMixture.name,
                "custom_options": {"num_components": 3}}
        },
        stop={"timesteps_total": 1000}
    )
