from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.tf_ops import make_tf_callable

from toolbox.dies.appo_impl.constants import *

tf = try_import_tf()


class TargetNetworkMixin:
    """This class implement the DELAY_UPDATE mechanism. Allowing:
    1. delayed update the targets networks of each policy.
    2. allowed fetches of action distribution of the target network of each
    policy.

    Note that this Mixin is with policy. That is to say, the target network
    of each policy is maintain by their own. After each training iteration, all
    policy will update their own target network.
    """

    def __init__(self, obs_space, action_space, config):
        assert config[DELAY_UPDATE]

        # Build the target network of this policy.
        _, logit_dim = ModelCatalog.get_action_dist(
            action_space, config["model"]
        )
        self.target_model = ModelCatalog.get_model_v2(
            obs_space,
            action_space,
            logit_dim,
            config["model"],
            name="target_func",
            framework="tf"
        )
        self.model_vars = self.model.variables()
        self.target_model_vars = self.target_model.variables()

        self.get_session().run(
            tf.variables_initializer(self.target_model_vars))

        # Here is the delayed update mechanism.
        self.tau_value = config.get("tau")
        self.tau = tf.placeholder(tf.float32, (), name="tau")
        assign_ops = []
        assert len(self.model_vars) == len(self.target_model_vars)
        for var, var_target in zip(self.model_vars, self.target_model_vars):
            assign_ops.append(
                var_target.
                    assign(self.tau * var + (1.0 - self.tau) * var_target)
            )
        self.update_target_expr = tf.group(*assign_ops)

        @make_tf_callable(self.get_session(), True)
        def compute_clone_network_logits(ob):
            feed_dict = {
                SampleBatch.CUR_OBS: tf.convert_to_tensor(ob),
                "is_training": tf.convert_to_tensor(False)
            }
            model_out, _ = self.target_model(feed_dict)
            return model_out

        self._compute_clone_network_logits = compute_clone_network_logits

    def update_target_network(self, tau=None):
        """Delayed update the target network."""
        tau = tau or self.tau_value
        return self.get_session().run(
            self.update_target_expr, feed_dict={self.tau: tau}
        )

    def update_target(self, tau=None):
        import warnings
        warnings.warn(
            "Please use update_target_network! Current update_target function "
            "is deprecated.",
            DeprecationWarning)
        return self.update_target_network(tau)
