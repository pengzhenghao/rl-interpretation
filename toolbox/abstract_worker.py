import logging
import time
from collections import OrderedDict

import ray
from ray.rllib.utils.memory import ray_get_and_free

from toolbox.utils import get_num_gpus, get_num_cpus

logger = logging.getLogger(__name__)


class WorkerBase:
    @classmethod
    def as_remote(cls,
                  num_cpus=None,
                  num_gpus=None,
                  memory=None,
                  object_store_memory=None,
                  resources=None):
        return ray.remote(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            memory=memory,
            object_store_memory=object_store_memory,
            resources=resources)(cls)

    def run(self, *args, **kwargs):
        """You should implement this method."""
        raise NotImplementedError()

    def close(self):
        ray.actor.exit_actor()


class WorkerManagerBase:
    """
    Example usage:
        def replay(self, name, symbolic_agent, obs):
            symbolic_agent.clear()
            ...
            self.submit(name, symbolic_agent, obs)
    """

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
            self.worker_dict[i] = {
                'worker':
                    worker_class.as_remote(num_gpus=num_gpus,
                                           num_cpus=num_cpus).remote(),
                'obj':
                    None,
                'name':
                    None,
                'time':
                    None
            }
        self.ret_dict = OrderedDict()
        self.start_count = 0
        self.finish_count = 0
        self.now = self.start = time.time()
        self.total_num = total_num
        self.log_interval = log_interval
        self.print_string = print_string
        self.deleted = False
        self._pointer = None

    def submit(self, agent_name, *args, **kwargs):


        print("enter submit: ", args, kwargs)

        """You should call this function at the main entry of your manager"""
        agent_name = str(agent_name)
        current_worker = None
        for i, wd in self.worker_dict.items():
            if wd['obj'] is None:
                self._pointer = i
                current_worker = wd['worker']
        assert current_worker is not None, \
            "No available worker found! {} | {}".format(
                self.worker_dict, self.get_status()
            )
        oid = current_worker.run.remote(*args, **kwargs)
        self.postprocess(agent_name, oid)

    def get_result(self):
        assert not self.deleted, self.error_string
        self._collect(force=True)
        # The result should be pickleable!
        for w_info in self.worker_dict.values():
            w_info['worker'].close.remote()
        self.worker_dict.clear()
        self.deleted = True

        return self.ret_dict

    def get_result_from_memory(self):
        return self.ret_dict

    def parse_result(self, result):
        """This function provide the string for printing."""
        return ""

    def get_status(self, force_wait=False, at_end=False):
        assert not self.deleted, self.error_string
        obj_list = [
            wd['obj'] for wd in self.worker_dict.values()
            if wd['obj'] is not None
        ]
        assert len(obj_list) <= self.num_workers
        if (not force_wait) or at_end:
            finished, pending = ray.wait(
                obj_list, len(obj_list), timeout=None if at_end else 0
            )
        else:
            finished, pending = ray.wait(obj_list, 1)
        return finished, pending

    def postprocess(self, name, obj_id):
        assert not self.deleted, self.error_string
        self.start_count += 1
        if self.start_count % self.log_interval == 0:
            print(
                "[{}/{}] (Task {:.2f}s|Total {:.2f}s) Start {}: {}!".format(
                    self.start_count, self.total_num, 0.0,
                    time.time() - self.start, self.print_string, name
                )
            )
            self.now = time.time()

        self.worker_dict[self._pointer]['name'] = name
        self.worker_dict[self._pointer]['obj'] = obj_id
        self.worker_dict[self._pointer]['time'] = time.time()

        self._collect()

    def _collect(self, force=False):
        """Update version of _collect only take one."""
        assert not self.deleted, self.error_string

        # If at the end.
        if force:
            finished, pending = self.get_status(at_end=True)
            assert len(pending) == 0
            self._get_object_list(finished)
            return

        # If at the beginning.
        finished, pending = self.get_status()
        if (len(finished) == 0) and \
                (len(pending) + len(finished) < self.num_workers):
            return

        if len(finished) == 0:
            # Load at least one result.
            finished, pending = self.get_status(force_wait=True)
            assert len(finished) >= 1
        self._get_object_list(finished)

    def _get_object_list(self, obj_list):
        for object_id in obj_list:
            ret = ray_get_and_free(object_id)
            name = None
            for worker_index, worker_info in self.worker_dict.items():
                if worker_info['obj'] == object_id:
                    name = worker_info['name']
                    break
            if name is None:
                raise ValueError()

            self.ret_dict[name] = ret
            task_start_time = self.worker_dict[worker_index]['time']
            self.worker_dict[worker_index]['obj'] = None
            self.worker_dict[worker_index]['name'] = None
            self.worker_dict[worker_index]['time'] = None

            self.finish_count += 1
            if self.finish_count % self.log_interval == 0:
                print(ray.available_resources())
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

    error_string = "The get_result function should only be called once! If " \
                   "you really want to retrieve the data," \
                   " please call self.get_result_from_memory() !"
