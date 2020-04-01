"""
This code is copied from

https://github.com/ray-project/ray/blob/master/rllib/models/tf/fcnet_v2.py

"""
from __future__ import absolute_import, division, print_function

import logging
from collections import OrderedDict

import numpy as np
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy, \
    EntropyCoeffSchedule, KLCoeffMixin, LearningRateSchedule, \
    BEHAVIOUR_LOGITS, ValueNetworkMixin
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.misc import normc_initializer, get_activation_fn
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import try_import_tf
from toolbox.utils import merge_dicts
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.backend import set_session

# from toolbox.modified_rllib.trainer_template import build_trainer

tf = try_import_tf()

logger = logging.getLogger(__name__)


# from toolbox.modified_rllib.tf_modelv2 import TFModelV2
class MultiplyMaskLayer(Layer):
    def __init__(self, output_dim, name, mask_mode, **kwargs):
        self.output_dim = output_dim
        self.mask_mode = mask_mode
        assert mask_mode in ['multiply', 'add']
        super(MultiplyMaskLayer, self).__init__(**kwargs)
        self.kernel = self.add_variable(
            name=name,
            shape=[
                output_dim,
            ],
            initializer='ones',
            trainable=False
        )

    def call(self, x, **kwargs):
        if self.mask_mode == 'add':
            ret = tf.add(x, self.kernel)
        else:  # self.mask_mode == "multiply"
            ret = tf.multiply(x, self.kernel)
        return ret

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim) + input_shape[1:]

    def get_kernel(self):
        return self.kernel


class FullyConnectedNetworkWithMask(TFModelV2):
    """Generic fully connected network implemented in ModelV2 API."""

    def __init__(
            self, obs_space, action_space, num_outputs, model_config, name
    ):
        super(FullyConnectedNetworkWithMask, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )

        activation_list = []
        self.activation_value = None

        mask_mode = model_config.get("custom_options")["mask_mode"]
        assert mask_mode in ['multiply', 'add']

        activation = get_activation_fn(model_config.get("fcnet_activation"))
        hiddens = model_config.get("fcnet_hiddens")
        no_final_linear = model_config.get("no_final_linear")
        vf_share_layers = model_config.get("vf_share_layers")

        inputs = tf.keras.layers.Input(
            shape=obs_space.shape, name="observations"
        )
        last_layer = inputs
        i = 1

        mask_placeholder_dict = OrderedDict()
        self.mask_layer_dict = OrderedDict()
        self.default_mask = OrderedDict()

        if no_final_linear:
            # the last layer is adjusted to be of size num_outputs
            for size in hiddens[:-1]:
                layer_name = "fc_{}".format(i)
                last_layer = tf.keras.layers.Dense(
                    size,
                    name=layer_name,
                    activation=activation,
                    kernel_initializer=normc_initializer(1.0)
                )(last_layer)

                # here is the multiplication
                mask_name = "fc_{}_mask".format(i)
                mask_layer = MultiplyMaskLayer(
                    size, name=mask_name, mask_mode=mask_mode
                )
                last_layer = mask_layer(last_layer)
                mask_placeholder_dict[mask_name] = mask_layer.get_kernel()
                self.mask_layer_dict[mask_name] = mask_layer

                activation_list.append(last_layer)
                i += 1

            layer_out = tf.keras.layers.Dense(
                num_outputs,
                name="fc_out",
                activation=activation,
                kernel_initializer=normc_initializer(1.0)
            )(last_layer)
        else:
            # the last layer is a linear to size num_outputs
            for size in hiddens:
                layer_name = "fc_{}".format(i)
                last_layer = tf.keras.layers.Dense(
                    size,
                    name=layer_name,
                    activation=activation,
                    kernel_initializer=normc_initializer(1.0)
                )(last_layer)

                # here is the multiplication
                mask_name = "fc_{}_mask".format(i)
                mask_layer = MultiplyMaskLayer(
                    size, name=mask_name, mask_mode=mask_mode
                )
                last_layer = mask_layer(last_layer)
                mask_placeholder_dict[mask_name] = mask_layer.get_kernel()
                self.mask_layer_dict[mask_name] = mask_layer

                activation_list.append(last_layer)
                i += 1

            layer_out = tf.keras.layers.Dense(
                num_outputs,
                name="fc_out",
                activation=None,
                kernel_initializer=normc_initializer(0.01)
            )(last_layer)

        if not vf_share_layers:
            # build a parallel set of hidden layers for the value net
            last_layer = inputs
            i = 1
            for size in hiddens:
                last_layer = tf.keras.layers.Dense(
                    size,
                    name="fc_value_{}".format(i),
                    activation=activation,
                    kernel_initializer=normc_initializer(1.0)
                )(last_layer)
                i += 1

        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01)
        )(last_layer)

        self.mask_placeholder_dict = mask_placeholder_dict

        self.base_model = tf.keras.Model(
            inputs=inputs, outputs=[layer_out, value_out] + activation_list
        )
        # TODO we can add a flag to determine whether to return activation.

        self.register_variables(self.base_model.variables)
        self.register_variables(list(self.mask_placeholder_dict.values()))

        for name, layer in self.mask_layer_dict.items():
            self.default_mask[name] = layer.get_weights()

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out, *self.activation_value = self.base_model(
            input_dict["obs_flat"]
        )
        return model_out, state

    def set_mask(self):
        for name, val in self.default_mask.items():
            assert name in self.mask_layer_dict
            assert isinstance(val, list)
            self.mask_layer_dict[name].set_weights(val)

    def set_default(self, mask_dict):
        for name, val in mask_dict.items():
            assert name in self.mask_layer_dict
            assert name in self.default_mask
            assert isinstance(val, np.object)
            assert list(val.shape) == \
                   self.mask_placeholder_dict[name].shape.as_list(), \
                (val.shape, self.mask_placeholder_dict[name].shape)

            # the layer only have one 'weight' so wrap it by list.
            self.default_mask[name] = [val]

        self.set_mask()

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def activation(self):
        return self.activation_value

    def from_batch(self, train_batch, is_training=True):
        """Convenience function that calls this model with a tensor batch.

        All this does is unpack the tensor batch to call this model with the
        right input dict, state, and seq len arguments.
        """

        input_dict = {
            "obs": train_batch[SampleBatch.CUR_OBS],
            "is_training": is_training,
        }

        for name, ph in self.mask_placeholder_dict.items():
            input_dict[name] = ph

        if SampleBatch.PREV_ACTIONS in train_batch:
            input_dict["prev_actions"] = train_batch[SampleBatch.PREV_ACTIONS]
        if SampleBatch.PREV_REWARDS in train_batch:
            input_dict["prev_rewards"] = train_batch[SampleBatch.PREV_REWARDS]
        states = []
        i = 0
        while "state_in_{}".format(i) in train_batch:
            states.append(train_batch["state_in_{}".format(i)])
            i += 1
        return self.__call__(input_dict, states, train_batch.get("seq_lens"))


def vf_preds_and_logits_fetches_new(policy):
    """Adds value function and logits outputs to experience batches."""
    ret = {
        SampleBatch.VF_PREDS: policy.model.value_function(),
        BEHAVIOUR_LOGITS: policy.model.last_output(),
    }

    activation_tensors = policy.model.activation()
    activations = {
        "layer{}".format(i): t
        for i, t in enumerate(activation_tensors)
    }
    ret.update(activations)
    return ret


def register_fc_with_mask():
    ModelCatalog.register_custom_model(
        "fc_with_mask", FullyConnectedNetworkWithMask
    )


# register_fc_with_mask()


class AddMaskInfoMixinForPolicy(object):
    def get_mask_info(self):
        return self.get_mask()

    def get_mask(self):
        ret = OrderedDict()
        for name, tensor in \
                self.model.mask_placeholder_dict.items():
            ret[name] = tensor.shape.as_list()
        return ret

    def set_default(self, mask_dict):
        with self._sess.graph.as_default():
            set_session(self._sess)
            # This fix the bug that
            self.model.set_default(mask_dict)

        logger.debug(
            "[AddMaskInfoMixinForPolicy] Successfully set the mask for: ", [
                "{}: array shape {}, mean {:.4f}, std {:.4f}".format(
                    k, v.shape, v.mean(), v.std()
                ) for k, v in mask_dict.items()
            ]
        )


def setup_mixins(policy, obs_space, action_space, config):
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(
        policy, config["entropy_coeff"], config["entropy_coeff_schedule"]
    )
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])
    AddMaskInfoMixinForPolicy.__init__(policy)


fc_with_mask_model_config = {
    "model": {
        "custom_model": "fc_with_mask",
        "custom_options": {}
    }
}

ppo_agent_default_config_with_mask = merge_dicts(
    DEFAULT_CONFIG, fc_with_mask_model_config
)

PPOTFPolicyWithMask = PPOTFPolicy.with_updates(
    name="PPOTFPolicyWithMask",
    get_default_config=lambda: ppo_agent_default_config_with_mask,
    extra_action_fetches_fn=vf_preds_and_logits_fetches_new,
    before_loss_init=setup_mixins,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin, AddMaskInfoMixinForPolicy
    ]
)


class AddMaskInfoMixin(object):
    def get_mask_info(self):
        return self.get_mask()

    def get_mask(self):
        return self.get_policy().get_mask()

    def set_mask(self, mask_dict):
        # Check the input is correct.
        exist_mask = self.get_mask()
        for name, arr in mask_dict.items():
            assert name in exist_mask
            assert list(arr.shape) == exist_mask[name]

        self.get_policy().set_default(mask_dict)
        if hasattr(self, "workers"):
            self.workers.foreach_worker(
                lambda w: w.get_policy().set_default(mask_dict)
            )

        logger.info(
            "Successfully set mask: {}".format(
                [
                    "layer: {}, shape: {}, mean {:.4f}, std {:.4f}.".format(
                        name, arr.shape, arr.mean(), arr.std()
                    ) for name, arr in mask_dict.items()
                ]
            )
        )
        print(
            "Successfully set mask: {}".format(
                [
                    "layer: {}, shape: {}, mean {:.4f}, std {:.4f}.".format(
                        name, arr.shape, arr.mean(), arr.std()
                    ) for name, arr in mask_dict.items()
                ]
            )
        )


# PPOTrainer.with_updates

PPOAgentWithMask = PPOTrainer.with_updates(
    name="PPOWithMask",
    default_config=ppo_agent_default_config_with_mask,
    default_policy=PPOTFPolicyWithMask,
    mixins=[AddMaskInfoMixin]
)
