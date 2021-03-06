from __future__ import absolute_import, division, print_function, \
    absolute_import, division, print_function

import copy
import logging
import os
import pickle

from ray.rllib.agents import Trainer
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.utils import try_import_tf

from toolbox.env.env_maker import get_env_maker
from toolbox.modified_rllib.agent_with_activation import (
    PPOAgentWithActivation, fc_with_activation_model_config,
    register_fc_with_activation
)
from toolbox.modified_rllib.agent_with_mask import (
    PPOAgentWithMask, register_fc_with_mask, PPOTFPolicyWithMask,
    ppo_agent_default_config_with_mask
)
from toolbox.utils import merge_dicts, has_gpu

logger = logging.getLogger(__name__)


def build_config(
        ckpt,
        extra_config=None,
        is_es_agent=False,
        change_model=None,
        use_activation_model=True
):
    if extra_config is None:
        extra_config = {}
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
                old_config = pickle.load(f)
                old_config.update(copy.deepcopy(config))
                config = copy.deepcopy(old_config)
    if "num_workers" in config:
        config["num_workers"] = min(1, config["num_workers"])
    if is_es_agent or (not use_activation_model):
        args_config = {}
    else:
        args_config = {"model": fc_with_activation_model_config}
    if has_gpu():
        args_config.update({"num_gpus_per_worker": 0.1})
    config = merge_dicts(config, args_config)
    config = merge_dicts(config, extra_config)
    if is_es_agent:
        config['num_workers'] = 1
        config['num_gpus_per_worker'] = 0
        config["num_gpus"] = 0
    if change_model:
        assert isinstance(change_model, str)
        config['model']['custom_model'] = change_model
    return config


def _restore(
        agent_type,
        run_name,
        ckpt,
        env_name,
        extra_config=None,
        existing_agent=None
):
    assert isinstance(agent_type, str) or issubclass(agent_type, Trainer)
    if existing_agent is not None:
        agent = existing_agent
    else:
        change_model = None
        use_activation_model = False
        if agent_type == "PPOAgentWithActivation":
            cls = PPOAgentWithActivation
            change_model = "fc_with_activation"
            use_activation_model = True
        elif agent_type == "PPOAgentWithMask":
            cls = PPOAgentWithMask
            change_model = "fc_with_mask"
            use_activation_model = True
        elif (not isinstance(agent_type, str)) and issubclass(agent_type,
                                                              Trainer):
            cls = agent_type
        else:
            cls = get_agent_class(run_name)
        is_es_agent = run_name == "ES"
        config = build_config(
            ckpt, extra_config, is_es_agent, change_model, use_activation_model
        )
        logger.info("The config of restored agent: ", config)
        agent = cls(env=env_name, config=config)
    if ckpt is not None:
        ckpt = os.path.abspath(os.path.expanduser(ckpt))  # Remove relative dir
        agent.restore(ckpt)
    return agent


def restore_agent_with_mask(
        run_name, ckpt, env_name, extra_config=None, existing_agent=None
):
    register_fc_with_mask()
    if extra_config is None:
        logger.info(
            "You do not specify the mask mode, "
            "we use 'multiply' mode as default."
        )
        extra_config = {"model": {"custom_options": {"mask_mode": "multiply"}}}
    return _restore(
        "PPOAgentWithMask", run_name, ckpt, env_name, extra_config,
        existing_agent
    )


def restore_policy_with_mask(run_name, ckpt, env_name, extra_config=None):
    tf = try_import_tf()
    Graph = tf.Graph

    assert run_name == "PPO"
    register_fc_with_mask()
    env = get_env_maker(env_name)()
    with Graph().as_default():
        # This is a workaround to avoid variable multiple init.
        p = PPOTFPolicyWithMask(
            env.observation_space, env.action_space,
            ppo_agent_default_config_with_mask
        )
        if ckpt is not None:
            path = os.path.abspath(os.path.expanduser(ckpt))
            wkload = pickle.load(open(path, 'rb'))['worker']
            state = pickle.loads(wkload)['state']['default_policy']
            p.set_state(state)
    return path


def restore_agent_with_activation(
        run_name, ckpt, env_name, extra_config=None, existing_agent=None
):
    register_fc_with_activation()
    return _restore(
        "PPOAgentWithActivation", run_name, ckpt, env_name, extra_config,
        existing_agent
    )


def restore_agent(
        run_name, ckpt, env_name, extra_config=None, existing_agent=None
):
    return _restore(
        run_name, run_name, ckpt, env_name, extra_config, existing_agent
    )
