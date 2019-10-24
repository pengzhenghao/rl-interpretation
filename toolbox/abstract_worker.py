import copy
import time
from collections import OrderedDict

import ray
from ray.rllib.utils.memory import ray_get_and_free

from toolbox.utils import get_num_gpus


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

        assert issubclass(worker_class, WorkerBase)
        self.worker_dict = OrderedDict()
        for i in range(num_workers):
            w = worker_class.as_remote(num_gpus=num_gpus).remote()
            self.worker_dict[i] = {'worker': w, 'obj': None, 'name': None}

        self.obj_name_dict = dict()
        self.ret_dict = OrderedDict()

        self.start_count = 0
        self.finish_count = 0
        self.now = self.start = time.time()
        self.total_num = total_num
        self.log_interval = log_interval
        self.print_string = print_string
        self.deleted = False
        self.pointer = None

    # Example usage:
    # def replay(self, name, symbolic_agent, obs):
    #     assert isinstance(symbolic_agent, SymbolicAgentBase)
    #     symbolic_agent.clear()
    #     oid = self.current_worker.replay.remote(symbolic_agent, obs)
    #     self._postprocess(name, oid)

    # @property
    def get_status(self, force=False):
        assert not self.deleted, self.error_string
        obj_list = [
            wd['obj'] for wd in self.worker_dict.values()
            if wd['obj'] is not None
        ]
        assert len(obj_list) <= self.num_workers
        if not force:
            finished, pending = ray.wait(obj_list, len(obj_list), timeout=0)
        else:
            # At least wait for one.
            finished, pending = ray.wait(obj_list, 1, None)
        return finished, pending

    @property
    def current_worker(self):
        assert not self.deleted, self.error_string
        # assert self.pointer is None, "self.pointer should be reset before!"
        """Assign an idle worker to current task."""
        for i, wd in self.worker_dict.items():
            if wd['obj'] is None:
                self.pointer = i
                return wd['worker']
        raise ValueError("No available worker found! {} | {}".format(
            self.worker_dict, self.get_status()
        ))

    def postprocess(self, name, obj_id):
        assert self.pointer is not None, \
            "You should call self.current_worker first!"
        assert not self.deleted, self.error_string
        self.start_count += 1
        if self.start_count % self.log_interval == 0:
            print(
                "[{}/{}] (+{:.2f}s/{:.2f}s) Start {}: {}!".format(
                    self.start_count, self.total_num,
                    time.time() - self.now,
                    time.time() - self.start, self.print_string, name
                )
            )
            self.now = time.time()

        self.worker_dict[self.pointer]['name'] = name
        self.worker_dict[self.pointer]['obj'] = obj_id
        self.obj_name_dict[obj_id] = {'index': self.pointer, 'name': name}

        self._collect()

        self.pointer = None

    def parse_result(self, result):
        """This function provide the string for printing."""
        return ""

    def _collect(self, force=False):
        """Update version of _collect only take one."""
        assert not self.deleted, self.error_string

        finished, pending = self.get_status()

        if (not force) and \
                ((len(pending) + len(finished) < self.num_workers)
                 and
                 (len(finished) == 0)):
            # test
            # assert self.current_worker is not None
            return

        # At least release one worker.
        finished, pending = self.get_status(force=True)

        if force:
            finished = finished + pending

        for oid in finished:
            ret = ray_get_and_free(oid)

            obj_info = self.obj_name_dict.pop(oid)
            name = obj_info['name']
            worker_index = obj_info['index']

            self.ret_dict[name] = ret

            self.worker_dict[worker_index]['obj'] = None
            self.worker_dict[worker_index]['name'] = None

            self.finish_count += 1
            if self.finish_count % self.log_interval == 0:
                print(
                    "[{}/{}] (+{:.2f}s/{:.2f}s) Finish {}: {}! {}".format(
                        self.finish_count, self.total_num,
                        time.time() - self.now,
                        time.time() - self.start, self.print_string, name,
                        self.parse_result(ret)
                    )
                )
                self.now = time.time()

    def get_result(self):
        assert not self.deleted, self.error_string
        self._collect(force=True)
        # The result should be pickleable!
        for w_info in self.worker_dict.values():
            w_info['worker'].close.remote()
        self.worker_dict.clear()
        self.obj_name_dict.clear()
        self.deleted = True
        return copy.deepcopy(self.ret_dict)

    def get_result_from_memory(self):
        return copy.deepcopy(self.ret_dict)

    error_string = "The get_result function should only be called once! If " \
                   "you really want to retrieve the data," \
                   " please call self.get_result_from_memory() !"
