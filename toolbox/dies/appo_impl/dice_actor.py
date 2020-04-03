"""
This file implements the logic of diversity computing
"""
import random

import numpy as np
import ray
from ray.rllib.optimizers.aso_aggregator import Aggregator
from ray.rllib.utils.actors import TaskPool
from ray.rllib.utils.annotations import override
from ray.rllib.utils.memory import ray_get_and_free


class DRAggregatorBase:
    """Aggregators should extend from this class."""

    def __init__(self, initial_weights_obj_id, remote_workers,
                 max_sample_requests_in_flight_per_worker, replay_proportion,
                 replay_buffer_num_slots, train_batch_size, sample_batch_size):
        """Initialize an aggregator.

        Arguments:
            initial_weights_obj_id (ObjectID): initial worker weights
            remote_workers (list): set of remote workers assigned to this agg
            max_sample_request_in_flight_per_worker (int): max queue size per
                worker
            replay_proportion (float): ratio of replay to sampled outputs
            replay_buffer_num_slots (int): max number of sample batches to
                store in the replay buffer
            train_batch_size (int): size of batches to learn on
            sample_batch_size (int): size of batches to sample from workers
        """

        self.broadcasted_weights = initial_weights_obj_id
        self.remote_workers = remote_workers
        self.sample_batch_size = sample_batch_size
        self.train_batch_size = train_batch_size

        if replay_proportion:
            if replay_buffer_num_slots * sample_batch_size <= train_batch_size:
                raise ValueError(
                    "Replay buffer size is too small to produce train, "
                    "please increase replay_buffer_num_slots.",
                    replay_buffer_num_slots, sample_batch_size,
                    train_batch_size)

        self.batch_buffer = []

        self.replay_proportion = replay_proportion
        self.replay_buffer_num_slots = replay_buffer_num_slots
        self.max_sample_requests_in_flight_per_worker = \
            max_sample_requests_in_flight_per_worker
        self.started = False
        self.replay_batches = []
        self.replay_index = 0
        self.num_sent_since_broadcast = 0
        self.num_weight_syncs = 0
        self.num_replayed = 0

    def start(self):
        # Kick off async background sampling
        self.sample_tasks = TaskPool()
        for ev in self.remote_workers:
            ev.set_weights.remote(self.broadcasted_weights)
            for _ in range(self.max_sample_requests_in_flight_per_worker):
                self.sample_tasks.add(ev, ev.sample.remote())
        self.started = True

    @override(Aggregator)
    def iter_train_batches(self, max_yield=999):
        """Iterate over train batches.

        Arguments:
            max_yield (int): Max number of batches to iterate over in this
                cycle. Setting this avoids iter_train_batches returning too
                much data at once.
        """
        assert self.started
        # ev is the rollout worker
        for ev, sample_batch in self._augment_with_replay(
                self.sample_tasks.completed_prefetch(
                    blocking_wait=True, max_yield=max_yield)):
            sample_batch.decompress_if_needed()
            self.batch_buffer.append(sample_batch)
            if sum(b.count
                   for b in self.batch_buffer) >= self.train_batch_size:
                if len(self.batch_buffer) == 1:
                    # make a defensive copy to avoid sharing plasma memory
                    # across multiple threads
                    train_batch = self.batch_buffer[0].copy()
                else:
                    train_batch = self.batch_buffer[0].concat_samples(
                        self.batch_buffer)
                yield train_batch

                self.batch_buffer = []

            # If the batch was replayed, skip the update below.
            if ev is None:
                continue

            # Put in replay buffer if enabled
            if self.replay_buffer_num_slots > 0:
                if len(self.replay_batches) < self.replay_buffer_num_slots:
                    self.replay_batches.append(sample_batch)
                else:
                    self.replay_batches[self.replay_index] = sample_batch
                    self.replay_index += 1
                    self.replay_index %= self.replay_buffer_num_slots

            # (DICE) Do not sync the weights
            # ev.set_weights.remote(self.broadcasted_weights)
            # self.num_weight_syncs += 1

            self.num_sent_since_broadcast += 1

            # Kick off another sample request
            self.sample_tasks.add(ev, ev.sample.remote())

    @override(Aggregator)
    def stats(self):
        return {
            "num_weight_syncs": self.num_weight_syncs,
            "num_steps_replayed": self.num_replayed,
        }

    @override(Aggregator)
    def reset(self, remote_workers):
        self.sample_tasks.reset_workers(remote_workers)

    def _augment_with_replay(self, sample_futures):
        def can_replay():
            num_needed = int(
                np.ceil(self.train_batch_size / self.sample_batch_size))
            return len(self.replay_batches) > num_needed

        for ev, sample_batch in sample_futures:
            sample_batch = ray_get_and_free(sample_batch)
            yield ev, sample_batch

            if can_replay():
                f = self.replay_proportion
                while random.random() < f:
                    f -= 1
                    replay_batch = random.choice(self.replay_batches)
                    self.num_replayed += replay_batch.count
                    yield None, replay_batch


class DRAggregator(DRAggregatorBase, Aggregator):
    """Simple single-threaded implementation of an Aggregator."""

    def __init__(self,
                 workers,
                 max_sample_requests_in_flight_per_worker=2,
                 replay_proportion=0.0,
                 replay_buffer_num_slots=0,
                 train_batch_size=500,
                 sample_batch_size=50,
                 broadcast_interval=5):
        self.workers = workers
        self.local_worker = workers.local_worker()
        self.broadcast_interval = broadcast_interval
        self.broadcast_new_weights()
        DRAggregatorBase.__init__(
            self, self.broadcasted_weights, self.workers.remote_workers(),
            max_sample_requests_in_flight_per_worker, replay_proportion,
            replay_buffer_num_slots, train_batch_size, sample_batch_size)

    @override(Aggregator)
    def broadcast_new_weights(self):
        self.broadcasted_weights = ray.put(self.local_worker.get_weights())
        self.num_sent_since_broadcast = 0

    @override(Aggregator)
    def should_broadcast(self):
        return self.num_sent_since_broadcast >= self.broadcast_interval
