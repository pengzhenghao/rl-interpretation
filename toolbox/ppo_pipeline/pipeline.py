import time

import ray
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.utils.timer import TimerStat

from toolbox.ppo_pipeline.actor import DRAggregator
from toolbox.ppo_pipeline.utils import WorkersConfig
from toolbox.ppo_pipeline.learner import AsyncLearnerThread, \
    SyncLearnerThread


class Pipeline:
    """Base class for pipeline"""

    @classmethod
    def as_remote(cls, num_cpus=None, num_gpus=None, memory=None,
                  object_store_memory=None, resources=None):
        """
        Usage: remote_instance = class.as_remote(num_cpu=1).remote()
        """
        return ray.remote(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            memory=memory,
            object_store_memory=object_store_memory,
            resources=resources
        )(cls)

    def __init__(self):
        pass

    def start(self):
        pass

    def end(self):
        pass


class PPOPipeline(Pipeline):

    def __init__(self,
                 worker_set_config,
                 train_batch_size=500,
                 sample_batch_size=50,
                 num_gpus=0,
                 replay_buffer_num_slots=0,
                 replay_proportion=0.0,
                 num_data_loader_buffers=1,
                 max_sample_requests_in_flight_per_worker=2,
                 broadcast_interval=1,
                 num_sgd_iter=1,
                 sgd_minibatch_size=1,
                 learner_queue_size=16,
                 learner_queue_timeout=300,
                 num_aggregation_workers=0,
                 shuffle_sequences=True,
                 sync_sampling=False,
                 minibatch_buffer_size=1,
                 _fake_gpus=False):

        super(PPOPipeline, self).__init__()

        assert isinstance(worker_set_config, WorkersConfig)

        self.workers = worker_set_config.make_workers()

        self.sync_sampling = sync_sampling

        # Setup learner
        if num_gpus > 1 or num_data_loader_buffers > 1:
            raise NotImplementedError()
        else:
            if self.sync_sampling:
                learner = SyncLearnerThread(
                    self.workers.local_worker(),
                    minibatch_buffer_size=minibatch_buffer_size,
                    num_sgd_iter=num_sgd_iter,
                    learner_queue_size=learner_queue_size,
                    learner_queue_timeout=learner_queue_timeout,
                    num_gpus=num_gpus,
                    sgd_batch_size=sgd_minibatch_size
                )
            else:
                learner = AsyncLearnerThread(
                    self.workers.local_worker(),
                    minibatch_buffer_size=minibatch_buffer_size,
                    num_sgd_iter=num_sgd_iter,
                    learner_queue_size=learner_queue_size,
                    learner_queue_timeout=learner_queue_timeout,
                )

            learner.start()
            self.learner = learner

        # Setup actor
        if num_aggregation_workers > 0:
            raise NotImplementedError()
        else:
            actor = DRAggregator(
                self.workers,
                replay_proportion=replay_proportion,
                max_sample_requests_in_flight_per_worker=(
                    max_sample_requests_in_flight_per_worker if not
                    self.sync_sampling else 1),
                replay_buffer_num_slots=replay_buffer_num_slots,
                train_batch_size=train_batch_size,
                sample_batch_size=sample_batch_size,
                broadcast_interval=broadcast_interval,
                sync_sampling=sync_sampling
            )
        self.actor = actor

        # Stats
        self._optimizer_step_timer = TimerStat()
        self._stats_start_time = time.time()
        self._last_stats_time = {}

        self.episode_history = []
        self.to_be_collected = []

