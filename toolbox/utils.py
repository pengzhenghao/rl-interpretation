import collections
import copy
import logging
import os
import time
import uuid
from collections import Mapping, Container
from sys import getsizeof

import numpy as np
import ray
from distro import linux_distribution
from ray.rllib.utils import merge_dicts

merge_dicts = merge_dicts


# from ray.internal.internal_api import unpin_object_data
# Update ray to 0.8.2 cause this function disappeared.


class DefaultMapping(collections.defaultdict):
    """default_factory now takes as an argument the missing key."""

    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value


def _is_centos():
    flag = linux_distribution(False)[0] == 'centos'
    if flag:
        # To make sure we are really talking about the same machine.
        assert linux_distribution(False)[1] == '7'
        assert linux_distribution(False)[2] == 'Core'
        assert linux_distribution(True)[0] == 'CentOS Linux'
        import pwd
        import os
        return str(pwd.getpwuid(os.getuid())[0])
    return flag


def initialize_ray(local_mode=False, num_gpus=None, test_mode=False, **kwargs):
    os.environ['OMP_NUM_THREADS'] = '1'

    ray.init(
        logging_level=logging.ERROR if not test_mode else logging.DEBUG,
        log_to_driver=test_mode,
        local_mode=local_mode,
        num_gpus=num_gpus,
        ignore_reinit_error=True,
        **kwargs
    )
    print("Successfully initialize Ray!")
    try:
        print("Available resources: ", ray.available_resources())
    except Exception:
        pass


def get_local_dir():
    """Deprecated"""
    return None


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
        num_gpus = (ray.available_resources()['GPU'] - 0.5) / num_workers
        if num_gpus >= 1:
            num_gpus = 1
    return num_gpus


def get_num_cpus(num_workers=None):
    num_cpus = 0
    if "CPU" not in ray.available_resources():
        return 0.5
    if has_gpu() and num_workers is not None:
        num_cpus = (ray.available_resources()['CPU'] - 0.5) / num_workers
        if num_cpus >= 1:
            num_cpus = 1
    return num_cpus


# ray_get_and_free is copied from ray
FREE_DELAY_S = 10.0
MAX_FREE_QUEUE_SIZE = 100
_last_free_time = 0.0
_to_free = []


def ray_get_and_free(object_ids):
    """Call ray.get and then queue the object ids for deletion.

    This function should be used whenever possible in RLlib, to optimize
    memory usage. The only exception is when an object_id is shared among
    multiple readers.

    Args:
        object_ids (ObjectID|List[ObjectID]): Object ids to fetch and free.

    Returns:
        The result of ray.get(object_ids).
    """

    global _last_free_time
    global _to_free

    # The copy here is really IMPORTANT! It remove the ref count of the object
    # and make it possible to be freed when the object_store of ray is limited.

    result = copy.deepcopy(ray.get(object_ids))  # no copy at origin

    if type(object_ids) is not list:
        object_ids = [object_ids]

    # for oid in object_ids:
    #     unpin_object_data(oid)

    _to_free.extend(object_ids)

    # batch calls to free to reduce overheads
    now = time.time()
    if (len(_to_free) > MAX_FREE_QUEUE_SIZE
            or now - _last_free_time > FREE_DELAY_S):
        ray.internal.free(_to_free)
        _to_free = []
        _last_free_time = now

    return result


def deep_getsizeof(o, ids=None):
    """Find the memory footprint of a Python object
    This is a recursive function that rills down a Python object graph
    like a dictionary holding nested ditionaries with lists of lists
    and tuples and sets.
    The sys.getsizeof function does a shallow size of only. It counts each
    object inside a container as pointer only regardless of how big it
    really is.
    :param o: the object
    :param ids:
    :return:
    """
    if ids is None:
        ids = set()
    d = deep_getsizeof
    if id(o) in ids:
        return 0

    r = getsizeof(o)
    ids.add(id(o))

    if isinstance(o, np.ndarray):
        return o.nbytes

    if isinstance(o, str):  # or isinstance(0, unicode):
        return r

    if isinstance(o, Mapping):
        return r + sum(d(k, ids) + d(v, ids) for k, v in o.items())

    if isinstance(o, Container):
        return r + sum(d(x, ids) for x in o)

    return r
