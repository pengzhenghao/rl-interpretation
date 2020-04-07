"""
Copied from ray/rllib/optimizers/async_samples_optimizer.py and Modified a lot
"""

import logging
import time
from collections import defaultdict

import numpy as np
import ray
from ray.rllib.optimizers.policy_optimizer import PolicyOptimizer
from ray.rllib.utils.annotations import override

logger = logging.getLogger(__name__)


# @ray.remote
# class ProgressMonitor:
#     def __init__(self, pipeline_ids):
#         self.progress = {pid: None for pid in pipeline_ids}
#
#     def _update(self, pipeline_id, progress):
#         print("HI! we are in monitor. Receive progress {} for pipeline {
#         }".format(
#             progress, pipeline_id
#         ))
#         assert pipeline_id in self.progress
#         self.progress[pipeline_id] = progress
#         print("Fuck you. in update, my id is: ", id(self))
#         print("HI! we are in monitor. The progress now after update: ",
#         self.progress)
#
#     def get_pipeline_callback(self, pipeline_id):
#         return lambda progress: self._update(pipeline_id, progress)
#
#     def get(self):
#         print("Fuck you. in get: my id is: ", id(self))
#         return copy.deepcopy(self.progress)


class Coordinator(PolicyOptimizer):
    """Main event loop of the IMPALA architecture.

    This class coordinates the data transfers between the learner thread
    and remote workers (IMPALA actors).

    TODO Current implementation do not support local mode.
    """

    def __init__(self, workers_config, make_pipeline, num_pipelines,
                 sync_sampling=False):

        use_less_workers = workers_config.make_workers(num_workers=0)

        # TODO how to deal with the workers here
        PolicyOptimizer.__init__(self, use_less_workers)

        self._stats_start_time = time.time()
        self._last_progress_dict = {}

        self.sync_sampling = sync_sampling

        pipeline_names = ["pipeline{}".format(i) for i in range(num_pipelines)]
        self.pipelines = {}
        self.pipeline_interfaces = {}
        for pipeline_id in pipeline_names:
            pipeline, interface = make_pipeline()
            self.pipelines[pipeline_id] = pipeline
            self.pipeline_interfaces[pipeline_id] = interface

        for pipeline in self.pipelines.values():
            pipeline.run.remote()

        logger.debug("===== Do you in sync sampling mode? {} =====".format(
            sync_sampling))

        # Stats
        self._stats_start_time = time.time()
        self._last_stats_time = {}
        self._last_stats_sum = {}

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

    def pull_progress(self, pipeline_id=None):
        if pipeline_id is None:
            return {
                pid: self.pull_progress(pid)
                for pid in self.pipeline_interfaces.keys()
            }
        else:
            assert pipeline_id in self.pipeline_interfaces
            return ray.get(
                self.pipeline_interfaces[pipeline_id].pull_progress.remote())

    def push_signal(self, signal, pipeline_id=None):
        if pipeline_id is None:
            for interface in self.pipeline_interfaces.values():
                interface.push_signal.remote(signal)
        else:
            assert pipeline_id in self.pipeline_interfaces
            self.pipeline_interfaces[pipeline_id].push_signal.remote(signal)

    @override(PolicyOptimizer)
    def step(self):
        progress_dict = self.pull_progress()
        self._last_progress_dict = progress_dict

        if all(progress is None for progress in progress_dict.values()):
            # The first iteration is not started.
            time.sleep(1)
            logger.debug("The first iter is not here. wait for 1 second.")
            return self.step()

        # We use the summation of all agents as these two stats
        sample_timesteps = sum(
            p["num_steps_sampled"]
            for p in progress_dict.values()
        )
        train_timesteps = sum(
            p["num_steps_trained"]
            for p in progress_dict.values()
        )

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
        # return {"timesteps_total": 0, "episode_reward_mean": 0}
        # TODO This function should run in each pipeline.

        # return_stats = {}
        #
        # episode_storage = {}
        #
        # for ws_id, workers in self.workers.items():
        #     episodes, self.to_be_collected[ws_id] = collect_episodes(
        #         workers.local_worker(),
        #         selected_workers or workers.remote_workers(),
        #         self.to_be_collected[ws_id],
        #         timeout_seconds=timeout_seconds)
        #     orig_episodes = list(episodes)
        #     missing = min_history - len(episodes)
        #     if missing > 0:
        #         episodes.extend(self.episode_history[ws_id][-missing:])
        #         assert len(episodes) <= min_history
        #     self.episode_history[ws_id].extend(orig_episodes)
        #     self.episode_history[ws_id] = self.episode_history[ws_id][
        #                                   -min_history:]
        #
        #     episode_storage[ws_id] = episodes
        #     res = summarize_episodes(episodes, orig_episodes)
        #     return_stats[ws_id] = res

        original_stat = PolicyOptimizer.stats(self)
        if self._last_progress_dict and (
                "episodes" in self._last_progress_dict["pipeline0"]):
            return_stats = parse_stats(
                {
                    pid: progress["collect_metrics"]
                    for pid, progress in self._last_progress_dict.items()
                }, {
                    pid: progress["episodes"]
                    for pid, progress in self._last_progress_dict.items()
                }
            )
            stats_list = [
                progress["stats"] for progress in
                self._last_progress_dict.values()
            ]
            ret_stat = wrap_dict_list(stats_list)
            learner_info = {p_id: progress["stats"]
                            for p_id, progress in
                            self._last_progress_dict.items()}

            ret_stat["learner"] = learner_info
            original_stat.update(ret_stat)
        else:
            return_stats = {}
        return_stats["info"] = original_stat
        print("Hi!!! This is the return stats: ", return_stats)
        return return_stats

    @override(PolicyOptimizer)
    def stop(self):
        self.push_signal("STOP")

    @override(PolicyOptimizer)
    def reset(self, remote_workers):
        # TODO use signal system to do this
        raise NotImplementedError()
        self.workers.reset(remote_workers)
        for aggregator in self.aggregator_set.values():
            aggregator.reset(remote_workers)

    # @override(PolicyOptimizer)
    # def stats(self):
    #     def timer_to_ms(timer):
    #         return round(1000 * timer.mean, 3)
    #
    #     stats_list = []
    #     learner_info = {}
    #
    #     for ws_id in self.aggregator_set.keys():
    #         aggregator = self.aggregator_set[ws_id]
    #         learner = self.learner_set[ws_id]
    #
    #         stats = aggregator.stats()
    #         stats.update(self.get_mean_stats_and_reset())
    #         stats["timing_breakdown"] = {
    #             "optimizer_step_time_ms": timer_to_ms(
    #                 self._optimizer_step_timer),
    #             "learner_grad_time_ms": timer_to_ms(learner.grad_timer),
    #             "learner_load_time_ms": timer_to_ms(learner.load_timer),
    #             "learner_load_wait_time_ms": timer_to_ms(
    #                 learner.load_wait_timer),
    #             "learner_dequeue_time_ms": timer_to_ms(learner.queue_timer),
    #         }
    #         stats["learner_queue"] = learner.learner_queue_size.stats()
    #         if learner.stats:
    #             learner_info["policy{}".format(ws_id)] = learner.stats
    #             if not self.sync_sampling:
    #                 learner_info["policy{}".format(ws_id)][
    #                 "train_timesteps"] \
    #                     = int(learner.stats[
    #                               "train_timesteps"] // learner.num_sgd_iter)
    #             learner_info["policy{}".format(ws_id)]["sample_timesteps"] = \
    #                 stats["sample_timesteps"]
    #             learner_info["policy{}".format(ws_id)][
    #             "training_iteration"] = \
    #                 int(stats["sample_timesteps"] // self.train_batch_size)
    #         stats.pop("sample_timesteps")
    #
    #         stats_list.append(stats)
    #
    #     ret_stat = wrap_dict_list(stats_list)
    #     ret_stat["learner"] = learner_info
    #     original_stat = PolicyOptimizer.stats(self)
    #     original_stat.update(ret_stat)
    #     return original_stat


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


def parse_stats(stat_dict, episode_storage):
    ret = {}

    default_key = ("agent0", "default_policy")

    # Assume each workerset only maintain one agent one policy, that's why we
    # use default_key
    policy_reward = {
        "policy{}".format(pid): [ep.agent_rewards[default_key] for ep in eps]
        for pid, eps in episode_storage.items()
    }

    rewards_max = {pid: max(rews) if rews else np.nan
                   for pid, rews in policy_reward.items()}
    rewards_mean = {pid: np.mean(rews) if rews else np.nan
                    for pid, rews in policy_reward.items()}
    rewards_min = {pid: min(rews) if rews else np.nan
                   for pid, rews in policy_reward.items()}

    flatten_rewards = [d for v in policy_reward.values() for d in v]

    ret["episode_reward_max"] = np.max(
        flatten_rewards) if flatten_rewards else np.nan
    ret["episode_reward_min"] = np.min(
        flatten_rewards) if flatten_rewards else np.nan
    ret["episode_reward_mean"] = np.mean(
        flatten_rewards) if flatten_rewards else np.nan

    ret["episode_len_mean"] = np.mean(
        [r["episode_len_mean"] for r in stat_dict.values()])
    ret["episodes_this_iter"] = sum(
        r["episodes_this_iter"] for r in stat_dict.values())

    ret["policy_reward_mean"] = rewards_mean
    ret["policy_reward_min"] = rewards_min
    ret["policy_reward_max"] = rewards_max

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
