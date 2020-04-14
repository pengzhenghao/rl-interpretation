import json
import os.path as osp

import gym
import numpy as np
from ray.rllib.models import ModelCatalog

from toolbox.action_distribution.mixture_gaussian import \
    register_gaussian_mixture
from toolbox.evaluate import restore_agent
from toolbox.evolution.modified_es import GaussianESTrainer
from toolbox.moges.fcnet_tanh import FullyConnectedNetworkTanh

register_gaussian_mixture()
ModelCatalog.register_custom_model(
    "fc_with_tanh", FullyConnectedNetworkTanh
)


def restore(ckpt, env_name="BipedalWalker-v2"):
    config = {}

    agent = restore_agent(
        GaussianESTrainer,
        ckpt,
        env_name,
        config
    )
    return agent


class MOGESAgent:

    def __init__(self, ckpt):
        with open(osp.join(osp.dirname(osp.dirname(ckpt)), "params.json"),
                  "rb") as f:
            config = json.load(f)

        config["num_workers"] = 0
        config["num_cpus_per_worker"] = 1
        config["num_cpus_for_driver"] = 1

        self.config = config
        self.config_env = gym.make(self.config["env"])
        self.action_dim = self.config_env.action_space.shape[0]
        self.k = config["model"]["custom_options"]["num_components"]
        self.std_mode = config["model"]["custom_options"]["std_norm"]
        self.expect_logit_length = (
            self.k * (1 + 2 * self.action_dim) if self.std_mode == "normal" else
            self.k * (1 + self.action_dim)
        )

        assert osp.exists(ckpt)
        agent = restore_agent(
            GaussianESTrainer,
            ckpt,
            "BipedalWalker-v2",
            config
        )
        self.agent = agent

        self._log_std = None

    def _check_shape(self):
        logits = self.get_logits(self.config_env.observation_space.sample())
        assert logits.shape == (1, self.expect_logit_length)

    def _get_std(self):
        if self._log_std == None:
            log_std = self.agent.get_policy().variables.get_weights()[
                "default_policy/learnable_log_std"]
            self._log_std = log_std
        return self._log_std

    def get_action(self, obs):
        assert self.config_env.observation_space.contaions(obs)
        return self.agent.get_policy().compute(obs)[0]

    def get_logits(self, obs):
        assert self.config_env.observation_space.contaions(obs)
        return self.agent.get_policy().compute_actions(obs)

    def get_dist(self, obs):
        logits = self.get_logits(obs)
        assert logits.ndim == 1

        if self.std_mode == "normal":
            mean, log_std, weight = np.split(
                logits,
                [self.action_dim * self.k, 2 * self.action_dim * self.k])
        else:
            mean, weight = np.split(
                logits,
                [self.action_dim * self.k])
            if self.std_mode == "zero":
                log_std = np.zeros_like(mean)
            elif self.std_mode == "free":
                log_std = self._get_std()
            else:
                raise NotImplementedError()
        assert len(weight) == self.k
        return mean, log_std, weight
