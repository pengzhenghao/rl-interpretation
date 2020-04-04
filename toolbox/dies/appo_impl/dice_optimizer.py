"""
Copied from ray/rllib/optimizers/async_samples_optimizer.py and Modified a lot
"""

import logging
import time
from collections import defaultdict

import numpy as np
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
from ray.rllib.optimizers.policy_optimizer import PolicyOptimizer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.timer import TimerStat

from toolbox.dies.appo_impl.dice_actor import DRAggregator
from toolbox.dies.appo_impl.dice_learner import LearnerThread
from toolbox.dies.appo_impl.dice_workers import SuperWorkerSet

logger = logging.getLogger(__name__)


class AsyncSamplesOptimizer(PolicyOptimizer):
    """Main event loop of the IMPALA architecture.

    This class coordinates the data transfers between the learner thread
    and remote workers (IMPALA actors).
    """

    def __init__(self,
                 workers,
                 train_batch_size=500,
                 sample_batch_size=50,
                 num_envs_per_worker=1,
                 num_gpus=0,
                 lr=0.0005,
                 replay_buffer_num_slots=0,
                 replay_proportion=0.0,
                 num_data_loader_buffers=1,
                 max_sample_requests_in_flight_per_worker=2,
                 broadcast_interval=1,
                 num_sgd_iter=1,
                 minibatch_buffer_size=1,
                 learner_queue_size=16,
                 learner_queue_timeout=300,
                 num_aggregation_workers=0,
                 _fake_gpus=False):
        PolicyOptimizer.__init__(self, workers)

        self._stats_start_time = time.time()
        self._last_stats_time = {}
        self._last_stats_sum = {}

        self.learner_set = {}
        self.aggregator_set = {}

        assert isinstance(workers, SuperWorkerSet)

        for ws_id, ws in workers.items():
            if num_gpus > 1 or num_data_loader_buffers > 1:
                # logger.info(
                #     "Enabling multi-GPU mode, {} GPUs, {} parallel
                #     loaders".format(
                #         num_gpus, num_data_loader_buffers))
                # if num_data_loader_buffers < minibatch_buffer_size:
                #     raise ValueError(
                #         "In multi-gpu mode you must have at least as many "
                #         "parallel data loader buffers as minibatch buffers: "
                #         "{} vs {}".format(num_data_loader_buffers,
                #                           minibatch_buffer_size))
                # self.learner = TFMultiGPULearner(
                #     self.workers.local_worker(),
                #     lr=lr,
                #     num_gpus=num_gpus,
                #     train_batch_size=train_batch_size,
                #     num_data_loader_buffers=num_data_loader_buffers,
                #     minibatch_buffer_size=minibatch_buffer_size,
                #     num_sgd_iter=num_sgd_iter,
                #     learner_queue_size=learner_queue_size,
                #     learner_queue_timeout=learner_queue_timeout,
                #     _fake_gpus=_fake_gpus)
                raise NotImplementedError()
            else:
                learner = LearnerThread(
                    ws.local_worker(),
                    minibatch_buffer_size=minibatch_buffer_size,
                    num_sgd_iter=num_sgd_iter,
                    learner_queue_size=learner_queue_size,
                    learner_queue_timeout=learner_queue_timeout)
            learner.start()
            self.learner_set[ws_id] = learner

            if num_aggregation_workers > 0:
                raise NotImplementedError()
                # self.aggregator = TreeAggregator(
                #     workers,
                #     num_aggregation_workers,
                #     replay_proportion=replay_proportion,
                #     max_sample_requests_in_flight_per_worker=(
                #         max_sample_requests_in_flight_per_worker),
                #     replay_buffer_num_slots=replay_buffer_num_slots,
                #     train_batch_size=train_batch_size,
                #     sample_batch_size=sample_batch_size,
                #     broadcast_interval=broadcast_interval)
            else:
                aggregator = DRAggregator(
                    ws,
                    replay_proportion=replay_proportion,
                    max_sample_requests_in_flight_per_worker=(
                        max_sample_requests_in_flight_per_worker),
                    replay_buffer_num_slots=replay_buffer_num_slots,
                    train_batch_size=train_batch_size,
                    sample_batch_size=sample_batch_size,
                    broadcast_interval=broadcast_interval)
            self.aggregator_set[ws_id] = aggregator

        # Stats
        self._optimizer_step_timer = TimerStat()
        self._stats_start_time = time.time()
        self._last_stats_time = {}

        self.episode_history = {ws_id: [] for ws_id, _ in self.workers.items()}
        self.to_be_collected = {ws_id: [] for ws_id, _ in self.workers.items()}

    def add_stat_val(self, key, val):
        if key not in self._last_stats_sum:
            self._last_stats_sum[key] = 0
            self._last_stats_time[key] = self._stats_start_time
        self._last_stats_sum[key] += val

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

    @override(PolicyOptimizer)
    def step(self):

        # workaround to start all sampling jobs
        if not self.aggregator_set[0].started:
            for aggregator in self.aggregator_set.values():
                aggregator.start()

        if len(self.workers.remote_workers()) == 0:
            raise ValueError("Config num_workers=0 means training will hang!")

        for l_id, learner in self.learner_set.items():
            assert learner.is_alive(), "{} is dead! ALl learners: {}.".format(
                l_id, self.learner_set.keys())

        with self._optimizer_step_timer:
            sample_timesteps, train_timesteps = self._step()

            # We use the summation of all agents as these two stats
            sample_timesteps = sum(sample_timesteps.values())
            train_timesteps = sum(train_timesteps.values())

        if sample_timesteps > 0:
            self.add_stat_val("sample_throughput", sample_timesteps)
        if train_timesteps > 0:
            self.add_stat_val("train_throughput", train_timesteps)

        self.num_steps_sampled += sample_timesteps
        self.num_steps_trained += train_timesteps

    @override(PolicyOptimizer)
    def collect_metrics(self,
                        timeout_seconds,
                        min_history=100,
                        selected_workers=None):
        """Returns worker and optimizer stats.

        Arguments:
            timeout_seconds (int): Max wait time for a worker before
                dropping its results. This usually indicates a hung worker.
            min_history (int): Min history length to smooth results over.
            selected_workers (list): Override the list of remote workers
                to collect metrics from.

        Returns:
            res (dict): A training result dict from worker metrics with
                `info` replaced with stats from self.
        """
        return_stats = {}
        for ws_id, workers in self.workers.items():
            episodes, self.to_be_collected[ws_id] = collect_episodes(
                workers.local_worker(),
                selected_workers or workers.remote_workers(),
                self.to_be_collected[ws_id],
                timeout_seconds=timeout_seconds)
            orig_episodes = list(episodes)
            missing = min_history - len(episodes)
            if missing > 0:
                episodes.extend(self.episode_history[ws_id][-missing:])
                assert len(episodes) <= min_history
            self.episode_history[ws_id].extend(orig_episodes)
            self.episode_history[ws_id] = self.episode_history[ws_id][
                                          -min_history:]
            res = summarize_episodes(episodes, orig_episodes)
            return_stats[ws_id] = res
        return_stats = parse_stats(return_stats)
        return_stats.update(info=self.stats())
        return_stats["info"]["learner_queue"].pop("size_quantiles")
        return return_stats

    @override(PolicyOptimizer)
    def stop(self):
        for learner in self.learner_set.values():
            learner.stopped = True

    @override(PolicyOptimizer)
    def reset(self, remote_workers):
        self.workers.reset(remote_workers)

        for aggregator in self.aggregator_set.values():
            aggregator.reset(remote_workers)

    @override(PolicyOptimizer)
    def stats(self):
        def timer_to_ms(timer):
            return round(1000 * timer.mean, 3)

        stats_list = []
        learner_info = {}

        for ws_id in self.aggregator_set.keys():
            aggregator = self.aggregator_set[ws_id]
            learner = self.learner_set[ws_id]

            stats = aggregator.stats()
            stats.update(self.get_mean_stats_and_reset())
            stats["timing_breakdown"] = {
                "optimizer_step_time_ms": timer_to_ms(
                    self._optimizer_step_timer),
                "learner_grad_time_ms": timer_to_ms(learner.grad_timer),
                "learner_load_time_ms": timer_to_ms(learner.load_timer),
                "learner_load_wait_time_ms": timer_to_ms(
                    learner.load_wait_timer),
                "learner_dequeue_time_ms": timer_to_ms(learner.queue_timer),
            }
            stats["learner_queue"] = learner.learner_queue_size.stats()
            if learner.stats:
                learner_info["policy{}".format(ws_id)] = learner.stats

            stats_list.append(stats)

        ret_stat = wrap_dict_list(stats_list)
        ret_stat["learner"] = learner_info
        original_stat = PolicyOptimizer.stats(self)
        original_stat.update(ret_stat)
        return original_stat

    def _step(self):
        sample_timesteps = {}
        train_timesteps = {}

        assert not hasattr(self, "aggregator")
        assert not hasattr(self, "learner")

        for (ws_id, aggregator), (ws_id2, learner) in zip(
                self.aggregator_set.items(),
                self.learner_set.items()
        ):
            assert ws_id == ws_id2

            sample_timesteps[ws_id] = 0
            train_timesteps[ws_id] = 0

            for train_batch in aggregator.iter_train_batches():
                sample_timesteps[ws_id] += train_batch.count
                learner.inqueue.put(train_batch)
                if (learner.weights_updated
                        and aggregator.should_broadcast()):
                    aggregator.broadcast_new_weights()

        for ws_id, learner in self.learner_set.items():
            while not learner.outqueue.empty():
                count = learner.outqueue.get()
                train_timesteps[ws_id] += count

        return sample_timesteps, train_timesteps


def wrap_dict_list(dict_list):
    assert isinstance(dict_list, list)
    data = defaultdict(list)
    for d in dict_list:
        for k, v in d.items():
            data[k].append(v)
    ret = {}
    for k, v in data.items():
        if all(np.isscalar(item) for item in v):
            # average the stats
            ret[k] = np.mean(v)
        else:
            if all(isinstance(item, dict) for item in v):
                # Nested dict
                ret[k] = wrap_dict_list(v)
            elif all(isinstance(item, list) for item in v):
                # Flatten list
                ret[k] = [data_point for item in v for data_point in item]
            else:
                raise ValueError("Data format incorrect! {}".format(v))
    return ret


def parse_stats(stat_dict):
    ret = {}
    rewards_max = {
        "policy{}".format(ws_id): r["episode_reward_max"]
        for ws_id, r in stat_dict.items()
        if np.isfinite(r["episode_reward_max"])
    }
    rewards_mean = {
        "policy{}".format(ws_id): r["episode_reward_mean"]
        for ws_id, r in stat_dict.items()
        if np.isfinite(r["episode_reward_mean"])
    }
    rewards_min = {
        "policy{}".format(ws_id): r["episode_reward_min"]
        for ws_id, r in stat_dict.items()
        if np.isfinite(r["episode_reward_min"])
    }
    ret["episode_reward_max"] = max(
        rewards_max.values()) if rewards_max else np.nan
    ret["episode_reward_min"] = min(
        rewards_min.values()) if rewards_min else np.nan
    ret["episode_reward_mean"] = np.mean(
        list(rewards_mean.values())) if rewards_mean else np.nan
    ret["episode_len_mean"] = np.mean(
        [r["episode_len_mean"] for r in stat_dict.values()])
    ret["episodes_this_iter"] = sum(
        r["episodes_this_iter"] for r in stat_dict.values())
    ret["policy_reward_mean"] = {
        "policy{}".format(ws_id): rewards_mean["policy{}".format(ws_id)]
        if "policy{}".format(ws_id) in rewards_mean else np.nan
        for ws_id in stat_dict.keys()
    }
    ret["policy_reward_min"] = {
        "policy{}".format(ws_id): rewards_min["policy{}".format(ws_id)]
        if "policy{}".format(ws_id) in rewards_min else np.nan
        for ws_id in stat_dict.keys()
    }
    ret["policy_reward_max"] = {
        "policy{}".format(ws_id): rewards_max["policy{}".format(ws_id)]
        if "policy{}".format(ws_id) in rewards_max else np.nan
        for ws_id in stat_dict.keys()
    }
    first_dict = next(iter(stat_dict.values()))
    ret["sampler_perf"] = wrap_dict_list(
        [r["sampler_perf"] for r in stat_dict.values()]
    )
    assert not first_dict["off_policy_estimator"]
    ret["off_policy_estimator"] = {}
    assert not first_dict["custom_metrics"]
    ret["custom_metrics"] = {}
    assert len(first_dict) == 11, "stat dict format not correct"
    return ret
