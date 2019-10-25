import collections
import logging
import uuid

import ray


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


def has_gpu():
    try:
        ret = "GPU" in ray.available_resources()
    except Exception:
        return False
    return ret


def get_num_gpus(num_workers=None):
    num_gpus = 0
    if has_gpu() and num_workers is not None:
        # A modest resource allocation
        num_gpus = (ray.available_resources()['GPU'] - 0.2) / num_workers / 3
        if num_gpus >= 1:
            num_gpus = 1
    return num_gpus
