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
        for w in self.workers:
            return w.set_global_vars(vars)

    def remote(self, vars):
        for w in self.workers:
            return w.set_global_vars.remote(vars)


class _RolloutWorkerSetContainer:
    def __init__(self, workers):
        self.workers = workers
        self.set_global_vars = _SetGlobalVarsFunc(workers)

    def foreach_trainable_policy(self, func):
        ret = []
        for w in self.workers:
            ret.append(w.foreach_trainable_policy(func))
        return ret


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
                [ws.local_worker() for ws in self._worker_sets.values()])
        else:
            self._check_worker_set_id(worker_set_id)
            return self._worker_sets[worker_set_id].local_worker()

    def remote_workers(self, worker_set_id=None):
        if worker_set_id is None:
            return [_RolloutWorkerSetContainer(ws.remote_workers())
                    for ws in self._worker_sets.values()]
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
