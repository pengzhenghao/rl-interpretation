import json
import os.path as osp

import gym
import numpy as np
import ray
from ray.rllib.models import ModelCatalog

from toolbox import initialize_ray
from toolbox.action_distribution.mixture_gaussian import \
    register_mixture_action_distribution
from toolbox.evaluate import restore_agent
from toolbox.evolution.modified_es import GaussianESTrainer
from toolbox.moges.fcnet_tanh import FullyConnectedNetworkTanh

register_mixture_action_distribution()
ModelCatalog.register_custom_model(
    "fc_with_tanh", FullyConnectedNetworkTanh
)


class MOGESAgent:
    def __init__(self, ckpt, existing_agent=None):
        if not ray.is_initialized():
            initialize_ray(num_gpus=0)

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
        if existing_agent is not None:
            assert isinstance(existing_agent, MOGESAgent)
            existing_agent = existing_agent.agent
        agent = restore_agent(
            GaussianESTrainer,
            ckpt,
            "BipedalWalker-v2",
            config,
            existing_agent=existing_agent
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
        assert self.config_env.observation_space.contains(obs)
        return self.agent.get_policy().compute(obs)[0]

    def get_logits(self, obs):
        assert self.config_env.observation_space.contains(obs)
        return self.agent.get_policy().compute_actions(obs)[0]

    def get_dist(self, obs):
        logits = self.get_logits(obs)
        assert logits.ndim == 1
        mean, log_std, weight = np.split(
            logits,
            [self.action_dim * self.k, 2 * self.action_dim * self.k])

        assert len(weight) == self.k
        return dict(
            mean=mean,
            log_std=log_std,
            weight=weight
        )
