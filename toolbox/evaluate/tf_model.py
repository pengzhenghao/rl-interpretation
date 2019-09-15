"""
This code is copied from

https://github.com/ray-project/ray/blob/master/rllib/models/tf/fcnet_v2.py

"""
from __future__ import absolute_import, division, print_function
import sys
sys.path.append("../")

from ray.rllib.models.tf.misc import normc_initializer, get_activation_fn
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import try_import_tf

tf = try_import_tf()
from ray.rllib.agents.ppo.ppo_policy import *
from ray.rllib.agents.ppo.ppo import *

from ray.rllib.models import ModelCatalog



# from ray.rllib.models.tf.tf_modelv2 import TFModelV2

# class MyModelClass(TFModelV2):
#     def __init__(self, obs_space, action_space, num_outputs, model_config,
#     name): ...
#     def forward(self, input_dict, state, seq_lens): ...
#     def value_function(self): ...


class FullyConnectedNetworkWithActivation(TFModelV2):
    """Generic fully connected network implemented in ModelV2 API."""

    def __init__(
            self, obs_space, action_space, num_outputs, model_config, name
    ):
        super(FullyConnectedNetworkWithActivation, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )

        activation_list = []
        self.activation_value = None

        activation = get_activation_fn(model_config.get("fcnet_activation"))
        hiddens = model_config.get("fcnet_hiddens")
        no_final_linear = model_config.get("no_final_linear")
        vf_share_layers = model_config.get("vf_share_layers")

        inputs = tf.keras.layers.Input(
            shape=obs_space.shape, name="observations"
        )
        last_layer = inputs
        i = 1

        if no_final_linear:
            # the last layer is adjusted to be of size num_outputs
            for size in hiddens[:-1]:
                last_layer = tf.keras.layers.Dense(
                    size,
                    name="fc_{}".format(i),
                    activation=activation,
                    kernel_initializer=normc_initializer(1.0)
                )(last_layer)
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
                last_layer = tf.keras.layers.Dense(
                    size,
                    name="fc_{}".format(i),
                    activation=activation,
                    kernel_initializer=normc_initializer(1.0)
                )(last_layer)
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

        self.base_model = tf.keras.Model(
            inputs, [layer_out, value_out] + activation_list
        )
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out, *self.activation_value = self.base_model(
            input_dict["obs_flat"]
        )
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def activation(self):
        return self.activation_value


def vf_preds_and_logits_fetches_new(policy):
    """Adds value function and logits outputs to experience batches."""
    ret = {
        SampleBatch.VF_PREDS: policy.value_function,
        BEHAVIOUR_LOGITS: policy.model_out,
    }
    activation_tensors = policy.model.activation()
    activations = {
        "layer{}".format(i): t
        for i, t in enumerate(activation_tensors)
    }
    ret.update(activations)
    return ret


model_config = {"custom_model": "fc_with_activation", "custom_options": {}}

ppo_default_config = ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG
ppo_default_config['model'].update(model_config)
PPOTFPolicyWithActivation = build_tf_policy(
    name="PPOTFPolicyWithActivation",
    get_default_config=lambda: ppo_default_config,
    loss_fn=ppo_surrogate_loss,
    stats_fn=kl_and_loss_stats,
    extra_action_fetches_fn=vf_preds_and_logits_fetches_new,
    postprocess_fn=postprocess_ppo_gae,
    gradients_fn=clip_gradients,
    before_init=setup_config,
    before_loss_init=setup_mixins,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin
    ]
)

ppo_agent_default_config = DEFAULT_CONFIG
ppo_agent_default_config['model'].update(model_config)
PPOAgentWithActivation = build_trainer(
    name="PPOWithActivation",
    default_config=ppo_agent_default_config,
    default_policy=PPOTFPolicyWithActivation,
    make_policy_optimizer=choose_policy_optimizer,
    validate_config=validate_config,
    after_optimizer_step=update_kl,
    after_train_result=warn_about_bad_reward_scales
)


def register():
    ModelCatalog.register_custom_model(
        "fc_with_activation", FullyConnectedNetworkWithActivation
    )


register()


def test_ppo():
    from toolbox.utils import initialize_ray
    initialize_ray(test_mode=True)
    po = PPOAgentWithActivation(env="BipedalWalker-v2", config=model_config)
    return po
