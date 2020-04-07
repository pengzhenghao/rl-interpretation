import logging
import time

import ray
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
from ray.rllib.utils.timer import TimerStat

from toolbox.ppo_pipeline.actor import DRAggregator
from toolbox.ppo_pipeline.learner import AsyncLearnerThread, \
    SyncLearnerThread
from toolbox.ppo_pipeline.utils import WorkersConfig

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

        self.num_steps_sampled = 0
        self.num_steps_trained = 0

        self._last_stats_time = {}
        self._last_stats_sum = {}

    def start(self):
        pass

    def end(self):
        pass

    def pull_signal(self):
        return ray.get(self.pipeline_interface.pull_signal.remote())

    def push_progress(self, progress):
        self.pipeline_interface.push_progress.remote(progress)

    def get_mean_stats_and_reset(self):
        now = time.time()
        mean_stats = {
            key: round(val / (now - self._last_stats_time[key]), 3)
            for key, val in self._last_stats_sum.items()
        }
        for key in self._last_stats_sum.keys():
            self._last_stats_sum[key] = 0
            self._last_stats_time[key] = time.time()
        return mean_stats

    def add_stat_val(self, key, val):
        if key not in self._last_stats_sum:
            self._last_stats_sum[key] = 0
            self._last_stats_time[key] = self._stats_start_time
        self._last_stats_sum[key] += val


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

        self.num_sgd_iter = num_sgd_iter
        self.train_batch_size = train_batch_size

    def run(self):
        while not self.stopped:
            self.step()

    def stop(self):
        self.workers.stop()

    def step(self):
        """
        Conduct a step of learning. Push the latest learning progress to
        interface.
        """

        signal = self.pull_signal()

        if signal == "STOP":
            logger.info("Receive signal: STOP. Terminate workers.")
            self.stop()
            return self.num_steps_sampled, self.num_steps_trained

        # workaround to start all sampling jobs
        if not self.actor.started:
            logger.debug("Kick off the sampling from optimizer.step")
            self.actor.start()

        # Check
        if len(self.workers.remote_workers()) == 0:
            raise ValueError("Config num_workers=0 means training will hang!")
        assert self.learner.is_alive()

        # Conduct a step of learning
        with self._optimizer_step_timer:
            sample_timesteps, train_timesteps = self._step()

        # Push the latest progress to interface
        self.num_steps_sampled += sample_timesteps
        self.num_steps_trained += train_timesteps
        progress = {}
        progress["collect_metrics"], progress["episodes"], progress["stats"] = \
            self.collect_metrics(timeout_seconds=180, min_history=100)
        progress["num_steps_sampled"] = progress["stats"]["num_steps_sampled"]
        progress["num_steps_trained"] = progress["stats"]["num_steps_trained"]
        self.push_progress(progress)

        return sample_timesteps, train_timesteps

    def _step(self):
        sample_timesteps = 0
        train_timesteps = 0

        # Collect data
        batch_count, step_count = _send_train_batch_to_learner(
            self.actor, self.learner)
        if self.sync_sampling:
            assert batch_count > 0 and step_count > 0
        sample_timesteps += step_count

        # Conduct learning
        batch_count, step_count = \
            _get_train_result_from_learner(self.learner, self.sync_sampling)
        if not self.sync_sampling:
            train_timesteps += int(
                step_count // self.learner.num_sgd_iter)
        else:
            train_timesteps += int(step_count)

        # Start sampling if necessary
        if self.sync_sampling:
            self.actor.start()

        return sample_timesteps, train_timesteps

    def collect_metrics(self, timeout_seconds, min_history=100,
                        selected_workers=None):
        episodes, self.to_be_collected = collect_episodes(
            self.workers.local_worker(),
            selected_workers or self.workers.remote_workers(),
            self.to_be_collected,
            timeout_seconds=timeout_seconds)
        orig_episodes = list(episodes)
        missing = min_history - len(episodes)
        if missing > 0:
            episodes.extend(self.episode_history[-missing:])
            assert len(episodes) <= min_history
        self.episode_history.extend(orig_episodes)
        self.episode_history = self.episode_history[-min_history:]
        res = summarize_episodes(episodes, orig_episodes)
        # res.update(info=self.stats)
        return res, episodes, self.stats()

    def stats(self):
        def timer_to_ms(timer):
            return round(1000 * timer.mean, 3)

        stats = self.actor.stats()
        stats.update(self.get_mean_stats_and_reset())
        stats["timing_breakdown"] = {
            "optimizer_step_time_ms": timer_to_ms(self._optimizer_step_timer),
            "learner_grad_time_ms": timer_to_ms(self.learner.grad_timer),
            "learner_load_time_ms": timer_to_ms(self.learner.load_timer),
            "learner_load_wait_time_ms": timer_to_ms(
                self.learner.load_wait_timer),
            "learner_dequeue_time_ms": timer_to_ms(self.learner.queue_timer),
        }
        stats["learner_queue"] = self.learner.learner_queue_size.stats()
        if self.learner.stats:
            stats["learner"] = self.learner.stats
        stats.update(
            num_steps_trained=self.num_steps_trained,
            num_steps_sampled=self.num_steps_sampled,
            training_iteration=int(
                self.num_steps_sampled // self.train_batch_size)
        )
        if not self.sync_sampling:
            stats["num_steps_trained"] /= self.num_sgd_iter
        return stats


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
