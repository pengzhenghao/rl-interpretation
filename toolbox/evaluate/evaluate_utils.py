from __future__ import absolute_import, division, print_function, \
    absolute_import, division, print_function

import os
import pickle

from ray.rllib.agents.registry import get_agent_class
from ray.tune.util import merge_dicts

from toolbox.evaluate.tf_model import PPOAgentWithActivation, model_config
from toolbox.utils import has_gpu
from toolbox.evaluate.tf_model import register


def build_config(ckpt, extra_config=None, is_es_agent=False):
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
    args_config = {} if is_es_agent else {"model": model_config}
    if has_gpu():
        args_config.update({"num_gpus_per_worker": 0.1})
    config = merge_dicts(config, args_config or {})
    config = merge_dicts(config, extra_config or {})
    return config


def restore_agent_with_activation(run_name, ckpt, env_name, extra_config=None):
    register()
    is_es_agent = run_name == "ES"
    # if config is None:
    config = build_config(ckpt, extra_config, is_es_agent)
    agent = PPOAgentWithActivation(env=env_name, config=config)
    if ckpt is not None:
        ckpt = os.path.abspath(os.path.expanduser(ckpt))  # Remove relative dir
        agent.restore(ckpt)
    return agent


def restore_agent(run_name, ckpt, env_name, extra_config=None):
    cls = get_agent_class(run_name)
    is_es_agent = run_name == "ES"
    config = build_config(ckpt, extra_config, is_es_agent)
    # This is a workaround
    if run_name == "ES":
        config["num_workers"] = 1
    agent = cls(env=env_name, config=config)
    if ckpt is not None:
        ckpt = os.path.abspath(os.path.expanduser(ckpt))  # Remove relative dir
        agent.restore(ckpt)
    return agent
