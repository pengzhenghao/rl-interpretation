import copy
import logging
import time
from collections import OrderedDict
import ray
# from ray.rllib.utils.memory import ray_get_and_free

###############################
# import ray
#
# # import ray
# # from ray.rllib.utils.memory import ray_get_and_free
#
FREE_DELAY_S = 10.0
MAX_FREE_QUEUE_SIZE = 100
# MAX_FREE_QUEUE_SIZE = 1
_last_free_time = 0.0
_to_free = []
_old_keys = set()

def ray_get_and_free(object_ids, max_free_queue_size=MAX_FREE_QUEUE_SIZE):
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

    global  _old_keys
    new_keys = copy.deepcopy(set([str(o) for o in ray.objects().keys()]))

    print('[ray get] difference: ', new_keys.symmetric_difference(_old_keys),
          len(new_keys), len(_old_keys), new_keys, _old_keys)

    _old_keys = new_keys

    print('[ray get] before get')
    print("[ray get] DEBUG: ", ray.worker.global_worker.plasma_client.debug_string())
    try:
        old_result = ray.get(object_ids)
        result = copy.deepcopy(old_result)
    except Exception as e:
        print('please stop here', ray.wait([object_ids], timeout=0))
        # print(ray.objects())
        raise e

    # old_keys = copy.deepcopy(set([str(o) for o in ray.objects().keys()]))


    if type(object_ids) is not list:
        object_ids = [object_ids]
    _to_free.extend(object_ids)

    print('[ray get] after get')
    # batch calls to free to reduce overheads
    now = time.time()
    if (len(_to_free) > max_free_queue_size
            or now - _last_free_time > FREE_DELAY_S):
        print('[ray get] start clean')
        ray.internal.free(_to_free)
        _to_free = []
        _last_free_time = now



    print('[ray get] exit')

    return result
###########################################

from toolbox.utils import get_num_gpus, get_num_cpus

logger = logging.getLogger(__name__)


class WorkerBase:
    @classmethod
    def as_remote(cls, num_cpus=None, num_gpus=None, resources=None):
        return ray.remote(
            num_cpus=num_cpus, num_gpus=num_gpus, resources=resources
        )(cls)

    def close(self):
        ray.actor.exit_actor()


class WorkerManagerBase:
    def __init__(
            self,
            num_workers,
            worker_class,
            total_num=None,
            log_interval=1,
            print_string=""
    ):
        self.num_workers = num_workers
        num_gpus = get_num_gpus(num_workers)
        num_cpus = get_num_cpus(num_workers)

        assert issubclass(worker_class, WorkerBase)
        self.worker_dict = OrderedDict()
        for i in range(num_workers):
            w = worker_class.as_remote(num_gpus=num_gpus,
                                       num_cpus=num_cpus).remote()
            self.worker_dict[i] = {'worker': w, 'obj': None, 'name': None,
                                   'time': None}

        # self.obj_name_dict = dict()
        self.ret_dict = OrderedDict()

        self.start_count = 0
        self.finish_count = 0
        self.now = self.start = time.time()
        self.total_num = total_num
        self.log_interval = log_interval
        self.print_string = print_string
        self.deleted = False
        self._pointer = None

        # self._accessed_obj = set()

    # Example usage:
    # def replay(self, name, symbolic_agent, obs):
    #     assert isinstance(symbolic_agent, SymbolicAgentBase)
    #     symbolic_agent.clear()
    #     oid = self.current_worker.replay.remote(symbolic_agent, obs)
    #     self._postprocess(name, oid)

    # @property
    def get_status(self, force_wait=False):
        assert not self.deleted, self.error_string
        obj_list = [
            wd['obj'] for wd in self.worker_dict.values()
            if wd['obj'] is not None
        ]
        assert len(obj_list) <= self.num_workers
        if not force_wait:
            finished, pending = ray.wait(obj_list, len(obj_list), timeout=0)
        else:
            # print("DEBUG use. Now we stuck at get_status: {}s".format(
            #     time.time() - self.start)
            # )
            # At least wait for one.
            finished, pending = ray.wait(obj_list, 1, None)
        return finished, pending

    @property
    def current_worker(self):
        assert not self.deleted, self.error_string
        # assert self._pointer is None, "self._pointer should be reset before!"
        """Assign an idle worker to current task."""
        for i, wd in self.worker_dict.items():
            if wd['obj'] is None:
                self._pointer = i
                return wd['worker']
        raise ValueError("No available worker found! {} | {}".format(
            self.worker_dict, self.get_status()
        ))

    def postprocess(self, name, obj_id):
        # assert self._pointer is not None, \
        #     "You should call self.current_worker first!"
        assert not self.deleted, self.error_string
        self.start_count += 1
        if self.start_count % self.log_interval == 0:
            print(
                "[{}/{}] (Task {:.2f}s|Total {:.2f}s) Start {}: {}!".format(
                    self.start_count, self.total_num,
                    0.0,
                    time.time() - self.start, self.print_string, name
                )
            )
            self.now = time.time()

        self.worker_dict[self._pointer]['name'] = name
        self.worker_dict[self._pointer]['obj'] = obj_id
        self.worker_dict[self._pointer]['time'] = time.time()
        # self.obj_name_dict[obj_id] = {'index': self._pointer, 'name': name}

        self._collect()

    def parse_result(self, result):
        """This function provide the string for printing."""
        return ""

    def _collect(self, force=False):
        """Update version of _collect only take one."""
        assert not self.deleted, self.error_string

        finished, pending = self.get_status()
        if (not force) and (len(finished) == 0) \
                and (len(pending) + len(finished) < self.num_workers):
            return

        while True:
            finished, pending = self.get_status(force_wait=True)
            # assert len(self.obj_name_dict) == len(finished) + len(pending)

            if (not finished) and (not pending):
                # Finish collection everything.
                assert force  # Now it should be get_result()
                break

            oid = finished[0]
            # assert oid not in self._accessed_obj
            # self._accessed_obj.add(str(oid))

            # if force:
            #     print(
            #         "In force _collect now {}, total {}. Curretn id {
            #         }".format(
            #             i, len(finished), oid
            #         ), ray.available_resources())
            #     assert oid in self.obj_name_dict.keys()
            ret = ray_get_and_free(oid, 2)

            # print("After collect oid index {}, we have get status: {
            # }".format(
            #     i, self.get_status()
            # ))

            found = False
            for worker_index, worker_info in self.worker_dict.items():
                if worker_info['obj'] == oid:
                    name = worker_info['name']
                    found = True
                    break
            if not found:
                raise ValueError()

            # obj_info = self.obj_name_dict.pop(oid)
            # name = obj_info['name']
            # worker_index = obj_info['index']

            self.ret_dict[name] = ret

            task_start_time = self.worker_dict[worker_index]['time']
            self.worker_dict[worker_index]['obj'] = None
            self.worker_dict[worker_index]['name'] = None
            self.worker_dict[worker_index]['time'] = None

            self.finish_count += 1
            if self.finish_count % self.log_interval == 0:
                print(
                    "[{}/{}] (Task {:.2f}s|Total {:.2f}s) Finish {}: {}! {}"
                    "".format(
                        self.finish_count, self.total_num,
                        time.time() - task_start_time,
                        time.time() - self.start, self.print_string, name,
                        self.parse_result(ret)
                    )
                )
                self.now = time.time()

            if not force:
                break

        # print('Quit _collect, force? {}. len obj_dict {}'.format(
        #     force, len(self.obj_name_dict)
        # ))

    def get_result(self):
        assert not self.deleted, self.error_string
        self._collect(force=True)
        # The result should be pickleable!
        for w_info in self.worker_dict.values():
            w_info['worker'].close.remote()
        self.worker_dict.clear()
        # self.obj_name_dict.clear()
        self.deleted = True
        return self.ret_dict

    def get_result_from_memory(self):
        return self.ret_dict

    error_string = "The get_result function should only be called once! If " \
                   "you really want to retrieve the data," \
                   " please call self.get_result_from_memory() !"


def test():
    """This is test codes for HEAVY MEMORY USAGE case. But just forget it."""
    # FIXME something wrong.
    from toolbox.utils import initialize_ray
    import numpy as np

    initialize_ray(test_mode=True)
    # initialize_ray(test_mode=True, redis_max_memory=2000000000)

    num = 100
    delay = 0

    class TestWorker(WorkerBase):
        def __init__(self):
            self.count = 0

        def count(self):
            time.sleep(delay)
            self.count += 1
            print(self.count, ray.cluster_resources())
            print("")
            # return copy.deepcopy((self.count, np.empty((20000000))))
            return self.count, np.empty((20000000))

    class TestManager(WorkerManagerBase):
        def __init__(self):
            super(TestManager, self).__init__(
                16, TestWorker, num, 1, 'test'
            )

        def count(self, index):
            oid = self.current_worker.count.remote()
            self.postprocess(index, oid)

    tm = TestManager()
    for i in range(num):
        tm.count(i)

    ret = tm.get_result()
    return ret


if __name__ == '__main__':
    ret = test()
    # import numpy as np
    #
    # ray.init()
    #
    # @ray.remote
    # class W:
    #     def f(self):
    #         return np.empty((10000000))
    #
    # oid_list = []
    #
    # worker = W.remote()
    # for i in range(100):
    #     oid = worker.f.remote()
    #     oid_list.append(oid)
    #
    # for i, oid in enumerate(oid_list):
    #     ray_get_and_free(oid)
    #     print(i, ray.available_resources())