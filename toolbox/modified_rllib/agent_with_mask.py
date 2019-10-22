"""
This code is copied from

https://github.com/ray-project/ray/blob/master/rllib/models/tf/fcnet_v2.py

"""
from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import numpy as np
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG, \
    validate_config, update_kl, \
    warn_about_bad_reward_scales, choose_policy_optimizer
from ray.rllib.agents.ppo.ppo_policy import kl_and_loss_stats, setup_config, \
    clip_gradients, EntropyCoeffSchedule, KLCoeffMixin, ppo_surrogate_loss, \
    LearningRateSchedule, BEHAVIOUR_LOGITS, postprocess_ppo_gae
from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.misc import normc_initializer, get_activation_fn
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import TFPolicy
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.annotations import override
from ray.rllib.utils.tf_ops import make_tf_callable
from ray.rllib.utils.tf_run_builder import TFRunBuilder
from ray.tune.util import merge_dicts
from tensorflow.keras.layers import Layer

from toolbox.modified_rllib.tf_policy_template import build_tf_policy
from toolbox.modified_rllib.trainer_template import build_trainer

tf = try_import_tf()


# from toolbox.modified_rllib.tf_modelv2 import TFModelV2
class MultiplyMaskLayer(Layer):
    def __init__(self, output_dim, name, default=None, **kwargs):
        self.output_dim = output_dim
        self.kernel = default

        super(MultiplyMaskLayer, self).__init__(**kwargs)

        if self.kernel is None:
            assert default is None
            self.kernel = self.add_variable(
                name=name,
                shape=[output_dim,],
                initializer='ones',
                trainable=False
            )

        # self.placeholder = tf.keras.layers.Input(
        #     tensor=self.kernel,
        #     name=name
        # )

        print("Finish init of multiplymasklayer")

    # def build(self, input_shape):
    #     Create a trainable weight variable for this layer.
    # output_shape = self.compute_output_shape(input_shape)
    # self.kernel = self.add_weight(name='kernel',
    #                               shape=(1,) + output_shape[1:],
    #                               initializer='ones',
    #                               trainable=True)
    # super(MyLayer, self).build(input_shape)  # Be sure to call this
    # somewhere!

    def call(self, x, **kwargs):
        # x, y = inp
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
                # mask_input = tf.keras.layers.Input(
                #     shape=size,
                #     name=mask_name,
                # )

                # mask_input = tf.Variable(
                #     tf.ones([1, size]),
                #     name=mask_name,
                #     trainable=False,
                #     # validate_shape=False
                # )

                mask_layer = MultiplyMaskLayer(
                    size, name=mask_name
                )

                last_layer = mask_layer(last_layer)

                # last_layer = tf.multiply(
                #     last_layer, mask_input, name="{}x{}".format(
                #     layer_name, mask_name)
                # )
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
                # mask_input = tf.keras.layers.Input(shape=size,
                # name=mask_name)
                # mask_input = tf.Variable(
                #     tf.ones([1, size]),
                #     name=mask_name,
                #     trainable=False,
                #     validate_shape=False
                # )
                mask_layer = MultiplyMaskLayer(
                    size, name=mask_name
                )

                last_layer = mask_layer(last_layer)


                # last_layer = tf.keras.layers.Multiply(
                #     name="{}x{}".format(layer_name, mask_name)
                # )(
                #     [last_layer, mask_input]
                # )

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

        # input_tensor = [inputs] + list(self.mask_placeholder_dict.values())
        input_tensor = inputs

        self.base_model = tf.keras.Model(
            inputs=input_tensor,
            # inputs=inputs,
            outputs=[layer_out, value_out] + activation_list
        )
        # TODO we can add a flag to determine whether to return activation.

        self.register_variables(self.base_model.variables)
        self.register_variables(list(self.mask_placeholder_dict.values()))

        for name, layer in self.mask_layer_dict.items():
            self.default_mask[name] = layer.get_weights()

    def forward(self, input_dict, state, seq_lens):
        # extra_input = [input_dict["obs_flat"]]

        # is_value_function = False
        # for name in self.mask_placeholder_dict.keys():
        #     if name not in input_dict:
        #         raise ValueError(
        #             "We are expecting: {} but you don't have {}. You have: {}"
        #             "".format(
        #                 self.mask_placeholder_dict.keys(), name,
        #                 input_dict.keys()
        #             ))
        #         # is_value_function = True
        #         # break
        #     extra_input.append(input_dict[name])

        # if is_value_function:
        #     batch_size = extra_input[0].shape.as_list()[0]
        #     extra_input.extend(self._build_ones(batch_size))

        model_out, self._value_out, *self.activation_value = self.base_model(
            input_dict["obs_flat"]
            # extra_input
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

from tensorflow.python.keras.backend import set_session


class ValueNetworkMixin_modified(object):
    def __init__(self, obs_space, action_space, config):

        # print("Enter ValueNetworkMixin_modified please stop here.")

        if config["use_gae"]:
            # print("init ValueNetworkMixin_modified without gae")
            @make_tf_callable(self.get_session())
            def value(ob, prev_action, prev_reward, *state):

                # print("Enter self._value, please stop here.")

                input_dict = {
                    SampleBatch.CUR_OBS: tf.convert_to_tensor([ob]),
                    SampleBatch.PREV_ACTIONS:
                        tf.convert_to_tensor([prev_action]),
                    SampleBatch.PREV_REWARDS:
                        tf.convert_to_tensor([prev_reward]),
                    "is_training": tf.convert_to_tensor(False)
                }

                # for name, tensor in self.model.mask_placeholder_dict.items():
                #     shape = [1] + tensor.shape.as_list()[1:]
                #     input_dict[name] = tf.ones(shape)

                model_out, _ = self.model(
                    input_dict, [tf.convert_to_tensor([s]) for s in state],
                    tf.convert_to_tensor([1])
                )
                return self.model.value_function()[0]

        else:

            @make_tf_callable(self.get_session())
            def value(ob, prev_action, prev_reward, *state):
                return tf.constant(0.0)

        self._value = value


# def postprocess_ppo_gae_deprecated(
#         policy, sample_batch, other_agent_batches=None, episode=None
# ):
#     """Adds the policy logits, VF preds, and advantages to the trajectory."""
#
#     completed = sample_batch["dones"][-1]
#     if completed:
#         last_r = 0.0
#     else:
#         next_state = []
#         for i in range(policy.num_state_tensors()):
#             next_state.append([sample_batch["state_out_{}".format(i)][-1]])
#
#         def value(ob, prev_action, prev_reward, *state):
#             print("Enter self._value, please stop here.")
#             input_dict = {
#                 SampleBatch.CUR_OBS: tf.convert_to_tensor([ob]),
#                 SampleBatch.PREV_ACTIONS: tf.convert_to_tensor([prev_action]),
#                 SampleBatch.PREV_REWARDS: tf.convert_to_tensor([prev_reward]),
#                 "is_training": tf.convert_to_tensor(False),
#                 "fc_1_mask": tf.zeros((1, 256)),
#                 "fc_2_mask": tf.zeros((1, 256)),
#             }
#             model_out, _ = policy.model.forward(
#                 input_dict, [tf.convert_to_tensor([s]) for s in state],
#                 tf.convert_to_tensor([1])
#             )
#             return policy.model.value_function()[0]
#
#         print("Enter modified postprocess_ppo_gae")
#         last_r = value(
#             sample_batch[SampleBatch.NEXT_OBS][-1],
#             sample_batch[SampleBatch.ACTIONS][-1],
#             sample_batch[SampleBatch.REWARDS][-1], *next_state
#         )
#     batch = compute_advantages(
#         sample_batch,
#         last_r,
#         policy.config["gamma"],
#         policy.config["lambda"],
#         use_gae=policy.config["use_gae"]
#     )
#     return batch


def register_fc_with_mask():
    ModelCatalog.register_custom_model(
        "fc_with_mask", FullyConnectedNetworkWithMask
    )


register_fc_with_mask()


# class AddDefaultMask(object):
#     def __init__(self):
#         self.default_mask_dict = None
#         self.batchsize_mask_dict = {}
#
#     def set_default_mask(self, mask_dict):
#         assert mask_dict is not None
#         assert mask_dict.keys() == self.model.mask_placeholder_dict.keys()
#         self.default_mask_dict = mask_dict
#         self.batchsize_mask_dict.clear()
#         print(
#             "Successfully set the default mask to: ",
#             {k: (v.mean(), v.std())
#              for k, v in mask_dict.items()}
#         )
#         self.model.set_default(mask_dict)
#         # self.model.set_mask(mask_dict)
#
#     def add_batchsize_mask_dict(self, batchsize):
#         if batchsize not in self.batchsize_mask_dict:
#             # assert self.default_mask_dict is not None
#
#             if self.default_mask_dict is None:
#                 mask_template = self.get_mask_info()
#                 self.set_default_mask(
#                     {
#                         # k: np.ones(shape[1:])
#                         k: np.ones(shape)
#                         for k, shape in mask_template.items()
#                     }
#                 )
#                 print(
#                     "Since you don't set the default mask, "
#                     "we set it to ones."
#                 )
#
#             mask_batch = {
#                 k: np.tile(v, (batchsize, 1))
#                 for k, v in self.default_mask_dict.items()
#             }
#             self.batchsize_mask_dict[batchsize] = mask_batch
#         return self.batchsize_mask_dict[batchsize]


# class ModifiedInputTensorMixin(object):
#     """Mixin for TFPolicy that adds entropy coeff decay."""
#
#     @override(TFPolicy)
#     def _build_compute_actions(
#             self,
#             builder,
#             obs_batch,
#             state_batches=None,
#             prev_action_batch=None,
#             prev_reward_batch=None,
#             episodes=None,
#             mask_batch=None,  # NEW!!
#     ):
#         state_batches = state_batches or []
#         if len(self._state_inputs) != len(state_batches):
#             raise ValueError(
#                 "Must pass in RNN state batches for placeholders {}, got {}".
#                     format(self._state_inputs, state_batches)
#             )
#         builder.add_feed_dict(self.extra_compute_action_feed_dict())
#         builder.add_feed_dict({self._obs_input: obs_batch})
#         if state_batches:
#             builder.add_feed_dict({self._seq_lens: np.ones(len(obs_batch))})
#         if self._prev_action_input is not None and \
#                 prev_action_batch is not None:
#             builder.add_feed_dict({self._prev_action_input: prev_action_batch})
#         if self._prev_reward_input is not None and \
#                 prev_reward_batch is not None:
#             builder.add_feed_dict({self._prev_reward_input: prev_reward_batch})
#         builder.add_feed_dict({self._is_training: False})
#
#         # # if mask_batch is None:
#         # #     mask_batch = self.add_batchsize_mask_dict(len(obs_batch))
#         #
#         # assert isinstance(mask_batch, dict), mask_batch
#         #
#         # for name, mask in mask_batch.items():
#         #     assert isinstance(mask, np.ndarray)
#         #     builder.add_feed_dict(
#         #         {self.model.mask_placeholder_dict[name]: mask}
#         #     )
#
#         # TODO
#         # if mask dict is not None, then we should update the model's
#         # default mask
#
#         # model_weights = self.model.get_weights()
#
#         builder.add_feed_dict(dict(zip(self._state_inputs, state_batches)))
#         fetches = builder.add_fetches(
#             [self._sampler] + self._state_outputs +
#             [self.extra_compute_action_fetches()]
#         )
#         return fetches[0], fetches[1:-1], fetches[-1]
#
#     @override(TFPolicy)
#     def compute_actions(
#             self,
#             obs_batch,
#             state_batches=None,
#             prev_action_batch=None,
#             prev_reward_batch=None,
#             info_batch=None,
#             episodes=None,
#             mask_batch=None,  # NEW!!!
#             **kwargs
#     ):
#         builder = TFRunBuilder(self._sess, "compute_actions")
#         fetches = self._build_compute_actions(
#             builder,
#             obs_batch,
#             state_batches,
#             prev_action_batch,
#             prev_reward_batch,
#             mask_batch=mask_batch
#         )
#         return builder.get(fetches)

class AddSetDefault(object):
    def set_default(self, mask_dict):
        with self._sess.graph.as_default():

            set_session(self._sess)

            # This fix the bug that
            self.model.set_default(mask_dict)

class AddMaskInfoMixinForPolicy(object):
    def get_mask_info(self):
        ret = OrderedDict()
        for name, tensor in \
                self.model.mask_placeholder_dict.items():
            ret[name] = tensor.shape.as_list()
        return ret


def setup_mixins(policy, obs_space, action_space, config):

    ValueNetworkMixin_modified.__init__(
        policy, obs_space, action_space, config
    )
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(
        policy, config["entropy_coeff"], config["entropy_coeff_schedule"]
    )
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])
    # ModifiedInputTensorMixin.__init__(policy)
    AddMaskInfoMixinForPolicy.__init__(policy)
    AddSetDefault.__init__(policy)


model_config = {
    "model": {
        "custom_model": "fc_with_mask",
        "custom_options": {}
    }
}

ppo_agent_default_config_with_mask = merge_dicts(DEFAULT_CONFIG, model_config)

# ppo_default_config_with_mask = ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG
# ppo_default_config_with_mask['model'].update(model_config)
PPOTFPolicyWithMask = build_tf_policy(
    name="PPOTFPolicyWithMask",
    get_default_config=lambda: ppo_agent_default_config_with_mask,
    loss_fn=ppo_surrogate_loss,
    stats_fn=kl_and_loss_stats,
    extra_action_fetches_fn=vf_preds_and_logits_fetches_new,
    postprocess_fn=postprocess_ppo_gae,
    gradients_fn=clip_gradients,
    before_init=setup_config,
    before_loss_init=setup_mixins,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin_modified,
        # ModifiedInputTensorMixin,
        # AddDefaultMask,
        AddMaskInfoMixinForPolicy,
        AddSetDefault
    ]
)


class AddMaskInfoMixin(object):
    def get_mask_info(self):
        ret = OrderedDict()
        for name, tensor in \
                self.get_policy().model.mask_placeholder_dict.items():
            ret[name] = tensor.shape.as_list()
        return ret


# ppo_agent_default_config_with_mask['model'].update(model_config)
PPOAgentWithMask = build_trainer(
    name="PPOWithMask",
    default_config=ppo_agent_default_config_with_mask,
    default_policy=PPOTFPolicyWithMask,

    make_policy_optimizer=choose_policy_optimizer,

    # For some reason, I can't generate the model with policy_optimizer,
    # So I just disable it.
    # PENGZHENGHAO
    # TODO YOU SHOULD ADD OPTIMIZER HERE!!!
    # make_policy_optimizer=None,
    validate_config=validate_config,
    after_optimizer_step=update_kl,
    after_train_result=warn_about_bad_reward_scales,
    mixins=[AddMaskInfoMixin]
)
