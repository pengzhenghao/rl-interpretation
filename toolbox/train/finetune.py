import copy
import time
from collections import OrderedDict

import ray

from toolbox.evaluate.symbolic_agent import SymbolicAgentBase
from toolbox.utils import has_gpu


class _RemoteSymbolicTrainWorker:
    @classmethod
    def as_remote(cls, num_cpus=None, num_gpus=None, resources=None):
        return ray.remote(
            num_cpus=num_cpus, num_gpus=num_gpus, resources=resources
        )(cls)

    def finetune(self, symbolic_agent, num_iters):
        agent = symbolic_agent.get()
        result_list = []
        for i in range(num_iters):
            result = agent.train()
            result_list.append(result)
        return result


class RemoteSymbolicTrainManager:
    def __init__(self, num_workers, total_num=None, log_interval=50):
        self.num_workers = num_workers
        num_gpus = 3.8 / num_workers if has_gpu() else 0
        self.workers = [
            _RemoteSymbolicTrainWorker.as_remote(num_gpus=num_gpus).remote()
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

    def train(self, index, symbolic_agent, num_iters):
        assert isinstance(symbolic_agent, SymbolicAgentBase)
        oid = self.workers[self.pointer].finetune.remote(symbolic_agent,
                                                         num_iters)
        self.obj_dict[index] = oid
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
                    "[{}/{}] (+{:.2f}s/{:.2f}s) Finish train: {}!".format(
                        self.finish_count, self.total_num,
                        time.time() - self.now,
                        time.time() - self.start, name
                    )
                )
                self.now = time.time()
        self.obj_dict.clear()

    def get_result(self):
        self._collect()
        return self.ret_dict
