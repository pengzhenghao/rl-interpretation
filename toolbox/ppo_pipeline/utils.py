import copy

import ray
from ray.rllib.evaluation.worker_set import WorkerSet


class WorkersConfig:
    def __init__(self, env_creator, policy, config, logdir):
        self.env_creator = env_creator
        self.policy = policy
        self.config = config
        self.logdir = logdir

    def make_workers(self, num_workers=None):
        if num_workers is None:
            num_workers = self.config["num_workers"]
        return WorkerSet(
            self.env_creator, self.policy, self.config, num_workers,
            self.logdir
        )

    def stop(self):
        pass


@ray.remote
class PipelineInterface:
    """This class serve as the communication interface between main program
    (Coordinator) and the pipeline.

    After each step of the pipeline, it update its progress via calling
    interface.push_progress.remote(progress) function. The master can
    retrieve the latest progress of this pipeline through
    ray.get(pipeline_interface.pull_progress.remote()).

    Coordinator can send signal to pipeline via calling
    pipeline_interface.push_signal.remote()
    Pipeline should retrieve the signal after each step.
    """

    def __init__(self):
        self._progress_to_coordinator = None
        self._signal_to_pipeline = None

    def push_progress(self, progress):
        self._progress_to_coordinator = progress

    def pull_progress(self):
        return copy.deepcopy(self._progress_to_coordinator)

    def push_signal(self, signal):
        self._signal_to_pipeline = signal

    def pull_signal(self):
        return copy.deepcopy(self._signal_to_pipeline)
