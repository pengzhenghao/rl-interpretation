import numpy as np
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.misc import normc_initializer, get_activation_fn
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import try_import_tf

tf = try_import_tf()

TANH_MODEL = "fc_with_tanh"


class FullyConnectedNetworkTanh(TFModelV2):
    """Generic fully connected network implemented in ModelV2 API."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):

        super(FullyConnectedNetworkTanh, self).__init__(
            obs_space, action_space, num_outputs, model_config, name)
        activation = get_activation_fn(model_config.get("fcnet_activation"))
        hiddens = model_config.get("fcnet_hiddens")
        no_final_linear = model_config.get("no_final_linear")
        vf_share_layers = model_config.get("vf_share_layers")

        k = model_config["custom_options"]["num_components"]
        action_length = (num_outputs // k - 1) // 2
        assert num_outputs % k == 0
        assert (num_outputs // k - 1) % 2 == 0
        insert_log_std = False
        if model_config["custom_options"].get("std_mode") == "free":
            # learnable parameters
            log_stds = tf.get_variable(
                name="learnable_log_std",
                shape=[k * action_length],
                initializer=tf.zeros_initializer
            )
            num_outputs -= k * action_length
            insert_log_std = True
        elif model_config["custom_options"].get("std_mode") == "zero":
            log_stds = tf.ones(name="log_std", shape=[k * action_length])
            num_outputs -= k * action_length
            insert_log_std = True

        # we are using obs_flat, so take the flattened shape as input
        inputs = tf.keras.layers.Input(
            shape=(np.product(obs_space.shape),), name="observations")
        last_layer = inputs
        i = 1

        if no_final_linear:
            raise NotImplementedError("no_final_linear should be set to False.")
            # # the last layer is adjusted to be of size num_outputs
            # for size in hiddens[:-1]:
            #     last_layer = tf.keras.layers.Dense(
            #         size,
            #         name="fc_{}".format(i),
            #         activation=activation,
            #         kernel_initializer=normc_initializer(1.0))(last_layer)
            #     i += 1
            # layer_out = tf.keras.layers.Dense(
            #     num_outputs,
            #     name="fc_out",
            #     activation=activation,
            #     kernel_initializer=normc_initializer(1.0))(last_layer)
        else:
            # the last layer is a linear to size num_outputs
            for size in hiddens:
                last_layer = tf.keras.layers.Dense(
                    size,
                    name="fc_{}".format(i),
                    activation=activation,
                    kernel_initializer=normc_initializer(1.0))(last_layer)
                i += 1
            layer_out = tf.keras.layers.Dense(
                num_outputs,
                name="fc_out",
                activation="tanh",  # <<== Here!
                kernel_initializer=normc_initializer(0.01))(last_layer)
            if model_config["custom_options"].get("std_mode") == "free":
                splits = tf.split(layer_out, [action_length * k, k], 1)
                with tf.control_dependencies([
                    tf.variables_initializer([log_stds])
                ]):
                    layer_out = tf.concat([
                        splits[0],
                        tf.broadcast_to(
                            log_stds, tf.shape(splits[0])),
                        splits[1]
                    ], axis=1)
            elif model_config["custom_options"].get("std_mode") == "zero":
                splits = tf.split(layer_out, [action_length * k, k], 1)
                layer_out = tf.concat([
                    splits[0],
                    tf.broadcast_to(
                        log_stds, tf.shape(splits[0])),
                    splits[1]
                ], axis=1)
        if not vf_share_layers:
            # build a parallel set of hidden layers for the value net
            last_layer = inputs
            i = 1
            for size in hiddens:
                last_layer = tf.keras.layers.Dense(
                    size,
                    name="fc_value_{}".format(i),
                    activation=activation,
                    kernel_initializer=normc_initializer(1.0))(last_layer)
                i += 1

        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(last_layer)

        self.base_model = tf.keras.Model(inputs, [layer_out, value_out])
        self.register_variables(self.base_model.variables)

        if model_config["custom_options"].get("std_mode") == "free":
            self.register_variables([log_stds])

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs_flat"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


def register_tanh_model():
    ModelCatalog.register_custom_model(
        TANH_MODEL, FullyConnectedNetworkTanh
    )
    print("Successfully registered tanh model!")
    return TANH_MODEL
