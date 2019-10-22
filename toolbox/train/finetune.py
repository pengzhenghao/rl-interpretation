from toolbox.utils import initialize_ray, has_gpu
from collections import OrderedDict
import time
import ray
import copy


class _RemoteSymbolicReplayWorker:
    @classmethod
    def as_remote(cls, num_cpus=None, num_gpus=None, resources=None):
        return ray.remote(
            num_cpus=num_cpus, num_gpus=num_gpus, resources=resources
        )(cls)

    def __init__(self):
        pass

    def finetune(self, symbolic_agent, num_iters):
        pass
        # if self.existing_agent is None:
        #     agent = symbolic_agent.get()['agent']
        #     self.existing_agent = agent
        # else:
        #     agent = symbolic_agent.get(self.existing_agent)['agent']
        # ret = agent_replay(agent, obs)
        # return ret


class RemoteSymbolicReplayManager:
    def __init__(self, num_workers, total_num=None, log_interval=50):
        self.num_workers = num_workers
        num_gpus = 3.8 / num_workers if has_gpu() else 0
        self.workers = [
            _RemoteSymbolicReplayWorker.as_remote(num_gpus=num_gpus).remote()
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

    def replay(self, index, symbolic_agent, obs):
        # assert isinstance(symbolic_agent, SymbolicAgentBase)
        oid = self.workers[self.pointer].replay.remote(symbolic_agent, obs)
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
                    "[{}/{}] (+{:.2f}s/{:.2f}s) Finish replay: {}!".format(
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


class FinetuneWorker:
    pass
