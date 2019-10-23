import copy
import time
from collections import OrderedDict

import ray

from toolbox.utils import get_num_gpus


class WorkerManagerBase:

    def __init__(self, num_workers, worker_class, total_num=None,
                 log_interval=1, print_string=""):
        self.num_workers = num_workers
        num_gpus = get_num_gpus(num_workers)

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

    # Example usage:
    # def replay(self, index, symbolic_agent, obs):
    #     assert isinstance(symbolic_agent, SymbolicAgentBase)
    #     symbolic_agent.clear()
    #     oid = self.current_worker.replay.remote(symbolic_agent, obs)
    #     self._postprocess(index, oid)

    @property
    def current_worker(self):
        return self.workers[self.pointer]

    def postprocess(self, index, obj_id):
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

    def _collect(self):
        for name, oid in self.obj_dict.items():
            ret = copy.deepcopy(ray.get(oid))
            self.ret_dict[name] = ret

            self.finish_count += 1
            if self.finish_count % self.log_interval == 0:
                print(
                    "[{}/{}] (+{:.2f}s/{:.2f}s) Finish {}: {}!".format(
                        self.finish_count, self.total_num,
                        time.time() - self.now,
                        time.time() - self.start, self.print_string, name
                    )
                )
                self.now = time.time()
        self.obj_dict.clear()

    def get_result(self):
        self._collect()
        # The result should be pickleable!
        return copy.deepcopy(self.ret_dict)
