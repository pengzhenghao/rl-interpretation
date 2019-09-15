from __future__ import absolute_import, division, print_function, \
    absolute_import, division, print_function

import collections
import logging
import os
import uuid

import ray
from gym.envs.box2d import BipedalWalker


class DefaultMapping(collections.defaultdict):
    """default_factory now takes as an argument the missing key."""

    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value


def initialize_ray(local_mode=False, num_gpus=0, test_mode=False):
    if not ray.is_initialized():
        ray.init(
            logging_level=logging.ERROR if not test_mode else logging.INFO,
            log_to_driver=test_mode,
            local_mode=local_mode,
            num_gpus=num_gpus
        )
        print("Sucessfully initialize Ray!")
    if not local_mode:
        print("Available resources: ", ray.available_resources())


def get_random_string():
    return str(uuid.uuid4())[:8]


def _get_num_iters_from_ckpt_name(ckpt):
    base_name = os.path.basename(ckpt)
    assert "-" in base_name
    assert base_name.startswith("checkpoint")
    num_iters = eval(base_name.split("-")[1])
    assert isinstance(num_iters, int)
    return num_iters


def build_env(useless=None):
    env = BipedalWalker()
    env.seed(0)
    return env


ENV_MAKER_LOOKUP = {"BipedalWalker-v2": build_env}


def has_gpu():
    try:
        ret = "GPU" in ray.available_resources()
    except Exception:
        return False
    return ret
