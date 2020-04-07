import time

import ray
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.utils.timer import TimerStat

from toolbox.ppo_pipeline.actor import DRAggregator
from toolbox.ppo_pipeline.learner import AsyncLearnerThread, \
    SyncLearnerThread


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
