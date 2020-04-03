"""
A set of WorkerSet. Each WorkerSet is responsible for one learning pipeline,
namely the training of a single agent.
"""

import logging

from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.utils import try_import_tf

tf = try_import_tf()

logger = logging.getLogger(__name__)


class _SetGlobalVarsFunc:
    def __init__(self, workers):
        self.workers = workers

    def __call__(self, vars):
        return {wid: w.set_global_vars(vars) for wid, w in self.workers.items()}

    def remote(self, vars):
        return {wid: w.set_global_vars.remote(vars) for wid, w in
                self.workers.items()}


class _RolloutWorkerSetContainer:
    def __init__(self, workers):
        self.workers = workers
        self.set_global_vars = _SetGlobalVarsFunc(workers)

    def foreach_trainable_policy(self, func):
        return {
            wid: w.foreach_trainable_policy(func)
            for wid, w in self.workers.items()
        }

    def get_policy(self, idx="default_policy"):
        return {wid: w.get_policy(idx) for wid, w in self.workers.items()}

    def save(self):
        raise NotImplementedError("In our design, this function should never "
                                  "be called. So please check!")
    #     return {wid: w.save() for wid, w in self.workers.items()}


class SuperWorkerSet:

    def __init__(self, num_sets, env_creator, policy, trainer_config=None,
                 num_workers_per_set=0, logdir=None, _setup=True):
        self._worker_sets = {}

        # What is policy? a class or an instance?

        for i in range(num_sets):
            self._worker_sets[i] = WorkerSet(
                env_creator, policy, trainer_config, num_workers_per_set,
                logdir, _setup)

    def items(self):
        return self._worker_sets.items()

    def add_workers(self, num_workers):
        for ws_id, ws in self._worker_sets.items():
            ws.add_workers(num_workers)
            logger.info(
                "Add {} workers to worker set {}".format(num_workers, ws_id))

    def _check_worker_set_id(self, worker_set_id):
        assert worker_set_id in self._worker_sets, \
            "Your input worker set id {} is not in {}.".format(
                worker_set_id, self._worker_sets.keys())

    def local_worker(self, worker_set_id=None):
        if worker_set_id is None:
            return _RolloutWorkerSetContainer(
                {wid: ws.local_worker() for wid, ws in self._worker_sets.items()
                 })
        else:
            self._check_worker_set_id(worker_set_id)
            return self._worker_sets[worker_set_id].local_worker()

    def remote_workers(self, worker_set_id=None):
        if worker_set_id is None:
            return [
                _RolloutWorkerSetContainer({
                    i: w for i, w in enumerate(ws.remote_workers())
                }) for ws in self._worker_sets.values()
            ]
        else:
            self._check_worker_set_id(worker_set_id)
            return self._worker_sets[worker_set_id].remote_workers()

    def reset(self, new_remote_workers=None, worker_set_id=None):
        if new_remote_workers is None:
            for ws in self._worker_sets.values():
                ws.reset(new_remote_workers)
        else:
            self._worker_sets[worker_set_id].reset(new_remote_workers)

    def stop(self, worker_set_id=None):
        if worker_set_id is None:
            for ws in self._worker_sets.values():
                ws.stop()
        else:
            self._check_worker_set_id(worker_set_id)
            self._worker_sets[worker_set_id].stop()

    def foreach_worker_with_index(self, func, worker_set_id=None):
        if worker_set_id is None:
            return {ws_id: ws.foreach_worker_with_index(func) for ws_id, ws in
                    self._worker_sets.items()}
        else:
            self._check_worker_set_id(worker_set_id)
            return self._worker_sets[worker_set_id].foreach_worker_with_index(
                func)

    def forwach_worker(self, func, worker_set_id=None):
        if worker_set_id is None:
            return {ws_id: ws.forwach_worker(func) for ws_id, ws in
                    self._worker_sets.items()}
        else:
            self._check_worker_set_id(worker_set_id)
            return self._worker_sets[worker_set_id].forwach_worker(func)
