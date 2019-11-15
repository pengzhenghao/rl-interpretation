import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution

tfd = tfp.distributions


class GaussianMixture(TFActionDistribution):
    name = "GaussianMixture"

    def __init__(self, inputs, model):
        self.k = model.model_config['custom_options']['num_components']
        input_length = inputs.shape.as_list()[1]
        action_length = int((input_length / self.k - 1) / 2)
        num_splits = [action_length] * int((input_length - self.k) /
        action_length) + [self.k]
        # splits = tf.split(inputs, num_splits, axis=1)

        splits = tf.split(inputs, int(input_length / self.k), axis=1)
        # self.log_stds = splits[:-1:2]

        self.log_stds = splits[1::2]
        self.stds = [tf.exp(log_std) for log_std in self.log_stds]
        self.means = splits[:-1:2]
        self.weight = splits[-1]

        self.mixture_dist = tfd.Categorical(logits=self.weight)
        # self.components_dist = tfd.MultivariateNormalDiag(
        self.components_dist = tfd.Normal(
            self.means,  # One for each component.
            self.stds)
        self.gaussian_mixture_model = tfd.MixtureSameFamily(
            mixture_distribution=self.mixture_dist,
            components_distribution=self.components_dist
        )
        TFActionDistribution.__init__(self, inputs, model)

    def logp(self, x):
        assert x.dtype == 'float32', x.dtype
        ret = self.gaussian_mixture_model.log_prob(x)
        # ret = tf.expand_dims(ret, 0)
        return ret

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
        GaussianMixture.name, GaussianMixture)


register_gaussian_mixture()

if __name__ == '__main__':
    from ray import tune
    from toolbox import initialize_ray
    from toolbox.action_distribution import PPOTrainerWithoutKL

    initialize_ray(test_mode=True, local_mode=True)

    tune.run(
        # "PPO",
        PPOTrainerWithoutKL,
        local_dir="/tmp/ray",
        name="DELETE_ME_TEST",
        config={
            "env": "BipedalWalker-v2",
            "model": {
                "custom_action_dist": GaussianMixture.name,
                "custom_options": {"num_components": 7}}
        },
        stop={"timesteps_total": 1000}
    )
