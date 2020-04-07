import logging
import time

import ray
from ray.rllib.utils.timer import TimerStat

from toolbox.ppo_pipeline.actor import DRAggregator
from toolbox.ppo_pipeline.learner import AsyncLearnerThread, \
    SyncLearnerThread
from toolbox.ppo_pipeline.utils import WorkersConfig, PipelineInterface

logger = logging.getLogger(__name__)


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

    def __init__(self, pipeline_interface):
        self.pipeline_interface = pipeline_interface
        pass

    def start(self):
        pass

    def end(self):
        pass

    def pull_signal(self):
        return ray.get(self.pipeline_interface.pull_signal.remote())

    def push_progress(self, progress):
        self.pipeline_interface.push_progress.remote(progress)


class PPOPipeline(Pipeline):

    def __init__(self,
                 worker_set_config,
                 pipeline_interface,
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

        super(PPOPipeline, self).__init__(pipeline_interface)

        assert isinstance(worker_set_config, WorkersConfig)

        self.workers = worker_set_config.make_workers()
        self.sync_sampling = sync_sampling
        # print("===!!! progress_callback_id", progress_callback)
        # self.submit_progress = progress_callback

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

        # TODO who will change this flag?
        self.stopped = False

        self._debug_time = time.time()
        self._debug_cnt = 0

    def run(self):
        while not self.stopped:
            self.step()

    def step(self):
        self.push_progress(
            "Hi! I am in pipeline at {} iteration! Elapse {} s.".format(
                self._debug_cnt, time.time() - self._debug_time
            ))
        self._debug_time = time.time()
        self._debug_cnt += 1
        time.sleep(0.5)
        return

        # workaround to start all sampling jobs
        # TODO you need to make sure even in sync mode the sampling is launch
        # automatically, after waiting for learner to finish one iter.
        if not self.actor.started:
            logger.debug("Kick off the sampling from optimizer.step")
            self.actor.start()

        if len(self.workers.remote_workers()) == 0:
            raise ValueError("Config num_workers=0 means training will hang!")

        assert self.learner.is_alive()

        with self._optimizer_step_timer:
            sample_timesteps, train_timesteps = self._step()

        return sample_timesteps, train_timesteps

        # if sample_timesteps > 0:
        #     self.add_stat_val("sample_throughput", sample_timesteps)
        # if train_timesteps > 0:
        #     self.add_stat_val("train_throughput", train_timesteps)
        #
        # self.num_steps_sampled += sample_timesteps
        # self.num_steps_trained += train_timesteps

    def _step(self):

        assert not hasattr(self, "aggregator")
        assert not hasattr(self, "learner")

        sample_timesteps = 0
        train_timesteps = 0

        batch_count, step_count = _send_train_batch_to_learner(
            self.actor, self.learner)
        if self.sync_sampling:
            assert batch_count > 0 and step_count > 0
        sample_timesteps += step_count

        batch_count, step_count = \
            _get_train_result_from_learner(self.learner, self.sync_sampling)
        if not self.sync_sampling:
            train_timesteps += int(
                step_count // self.learner.num_sgd_iter)
        else:
            train_timesteps += int(step_count)

        if self.sync_sampling:
            self.actor.start()

        return sample_timesteps, train_timesteps


def _send_train_batch_to_learner(aggregator, learner):
    batch_count = 0
    step_count = 0
    for train_batch in aggregator.iter_train_batches():
        batch_count += 1
        step_count += train_batch.count
        learner.inqueue.put(train_batch)
        if (learner.weights_updated and aggregator.should_broadcast()):
            aggregator.broadcast_new_weights()

    # TODO In sync mode, should we expect batch_count to always be one???
    return batch_count, step_count


def _get_train_result_from_learner(learner, sync_sampling):
    batch_count = 0
    step_count = 0
    learner_finished = False
    while True:
        if (not sync_sampling) and learner.outqueue.empty():
            break
        count = learner.outqueue.get(block=True)
        if count is None:
            # This only happen in sync mode when train batch is exhaust
            # trained!
            learner_finished = True
            assert learner.outqueue.empty()
        else:
            batch_count += 1
            step_count += count
        if (not sync_sampling) or learner_finished:
            break
    return batch_count, step_count
