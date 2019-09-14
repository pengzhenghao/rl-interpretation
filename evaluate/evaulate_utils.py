from __future__ import absolute_import, division, print_function, \
    absolute_import, division, print_function

import collections
import distutils
import logging
import os
import pickle
import subprocess
import tempfile
import time
import uuid
from math import floor

import cv2
import numpy as np
import ray
from PIL import Image
from gym import logger, error
from gym.envs.box2d import BipedalWalker
from ray.rllib.agents.registry import get_agent_class
from ray.tune.util import merge_dicts

from evaluate.tf_model import PPOAgentWithActivation, model_config


def build_config(ckpt, args_config, extra_config=None):
    config = {"log_level": "ERROR"}
    if ckpt is not None:
        ckpt = os.path.abspath(os.path.expanduser(ckpt))  # Remove relative dir
        # config = {"log_level": "ERROR"}
        # Load configuration from file
        config_dir = os.path.dirname(ckpt)
        config_path = os.path.join(config_dir, "params.pkl")
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, "../params.pkl")
        if os.path.exists(config_path):
            with open(config_path, "rb") as f:
                config.update(pickle.load(f))
    if "num_workers" in config:
        config["num_workers"] = min(1, config["num_workers"])
    config = merge_dicts(config, args_config or {})
    config = merge_dicts(config, extra_config or {})
    return config


def restore_agent_with_activation(run_name, ckpt, env_name, extra_config=None):
    # if config is None:
    if not has_gpu():
        args_config = {"model": model_config}
    else:
        args_config = {"model": model_config, "num_gpus_per_worker": 0.1}
    config = build_config(ckpt, args_config, extra_config)
    agent = PPOAgentWithActivation(env=env_name, config=config)
    if ckpt is not None:
        ckpt = os.path.abspath(os.path.expanduser(ckpt))  # Remove relative dir
        agent.restore(ckpt)
    return agent


def restore_agent(run_name, ckpt, env_name, extra_config=None):
    cls = get_agent_class(run_name)
    # if config is None:
    if not has_gpu():
        args_config = {}
    else:
        args_config = {"num_gpus_per_worker": 0.1}
    config = build_config(ckpt, args_config, extra_config)
    # This is a workaround
    if run_name == "ES":
        config["num_workers"] = 1
    agent = cls(env=env_name, config=config)
    if ckpt is not None:
        ckpt = os.path.abspath(os.path.expanduser(ckpt))  # Remove relative dir
        agent.restore(ckpt)
    return agent
