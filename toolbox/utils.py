from __future__ import absolute_import, division, print_function, \
    absolute_import, division, print_function

import collections
import logging
import os
import uuid

import ray

from toolbox.env.env_maker import build_bipedal_walker, ENV_MAKER_LOOKUP


class DefaultMapping(collections.defaultdict):
    """default_factory now takes as an argument the missing key."""

    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value


def initialize_ray(local_mode=False, num_gpus=0, test_mode=False, **kwargs):
    if not ray.is_initialized():
        ray.init(
            logging_level=logging.ERROR if not test_mode else logging.DEBUG,
            log_to_driver=test_mode,
            local_mode=local_mode,
            num_gpus=num_gpus,
            **kwargs
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


build_env = build_bipedal_walker

ENV_MAKER_LOOKUP = ENV_MAKER_LOOKUP


def has_gpu():
    try:
        ret = "GPU" in ray.available_resources()
    except Exception:
        return False
    return ret
