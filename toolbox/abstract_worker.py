import copy
import time
from collections import OrderedDict

import ray

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

    def __init__(self, num_workers, worker_class, total_num=None,
                 log_interval=1, print_string=""):
        self.num_workers = num_workers
        num_gpus = get_num_gpus(num_workers)

        assert issubclass(worker_class, WorkerBase)
        self.workers = [
            worker_class.as_remote(num_gpus=num_gpus).remote()
            for _ in range(num_workers)
        ]

        self.pointer = 0
        self.obj_dict = OrderedDict()
        self.ret_dict = OrderedDict()
        self.start_count = 0
        self.finish_count = 0
        self.now = self.start = time.time()
        self.total_num = total_num
        self.log_interval = log_interval
        self.print_string = print_string
        self.deleted = False

    # Example usage:
    # def replay(self, index, symbolic_agent, obs):
    #     assert isinstance(symbolic_agent, SymbolicAgentBase)
    #     symbolic_agent.clear()
    #     oid = self.current_worker.replay.remote(symbolic_agent, obs)
    #     self._postprocess(index, oid)

    @property
    def current_worker(self):
        assert not self.deleted
        return self.workers[self.pointer]

    def postprocess(self, index, obj_id):
        assert not self.deleted, self.error_string
        self.start_count += 1
        if self.start_count % self.log_interval == 0:
            print(
                "[{}/{}] (+{:.2f}s/{:.2f}s) Start {}: {}!".format(
                    self.start_count, self.total_num,
                    time.time() - self.now,
                    time.time() - self.start, self.print_string, index
                )
            )
            self.now = time.time()

        self.obj_dict[index] = obj_id
        self.pointer += 1
        if self.pointer == self.num_workers:
            self._collect()
            self.pointer = 0

    def parse_result(self, result):
        """This function provide the string for printing."""
        return ""

    def _collect(self):
        assert not self.deleted, self.error_string
        for name, oid in self.obj_dict.items():
            ret = copy.deepcopy(ray.get(oid))
            self.ret_dict[name] = ret

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
        self.obj_dict.clear()

    def get_result(self):
        assert not self.deleted, self.error_string
        self._collect()
        # The result should be pickleable!
        for w in self.workers:
            w.close.remote()
        self.workers.clear()
        self.deleted = True
        return copy.deepcopy(self.ret_dict)

    def get_result_from_memory(self):
        return copy.deepcopy(self.ret_dict)

    error_string = "The get_result function should only be called once! If you" \
                   "really want to retrieve the data," \
                   " please call self.get_result_from_memory() !"
