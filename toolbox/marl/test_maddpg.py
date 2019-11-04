from __future__ import absolute_import, division, print_function

import numpy as np
from gym.spaces import Tuple
from ray import tune
from ray.rllib.examples.twostep_game import TwoStepGame
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.timer import TimerStat
from ray.tune import register_env

from toolbox import initialize_ray
from toolbox.marl.maddpg import MADDPGTrainer


def _build_matrix(iterable, apply_function, default_value=0):
    """
    Copied from toolbox.interface.cross_agent_analysis
    """
    length = len(iterable)
    matrix = np.empty((length, length))
    matrix.fill(default_value)
    for i1 in range(length - 1):
        for i2 in range(i1, length):
            repr1 = iterable[i1]
            repr2 = iterable[i2]
            result = apply_function(repr1, repr2)
            matrix[i1, i2] = result
            matrix[i2, i1] = result
    return matrix


def test_maddpg_basic(extra_config=None, local_mode=True):
    grouping = {
        "group_1": [0, 1],
    }
    obs_space = Tuple(
        [
            TwoStepGame.observation_space,
            TwoStepGame.observation_space,
        ]
    )
    act_space = Tuple([
        TwoStepGame.action_space,
        TwoStepGame.action_space,
    ])
    register_env(
        "grouped_twostep", lambda config: TwoStepGame(config).
            with_agent_groups(grouping, obs_space=obs_space,
                              act_space=act_space)
    )

    config = {
        "learning_starts": 100,
        "env_config": {
            "actions_are_logits": True,
        },
        "multiagent": {
            "policies": {
                "pol1": (
                    None, TwoStepGame.observation_space,
                    TwoStepGame.action_space, {
                        "agent_id": 0,
                    }
                ),
                "pol2": (
                    None, TwoStepGame.observation_space,
                    TwoStepGame.action_space, {
                        "agent_id": 1,
                    }
                ),
            },
            "policy_mapping_fn": lambda x: "pol1" if x == 0 else "pol2",
        },
    }

    if isinstance(extra_config, dict):
        config.update(extra_config)

    initialize_ray(test_mode=True, local_mode=local_mode)
    tune.run(
        MADDPGTrainer,
        stop={
            "timesteps_total": 1000,
        },
        config=dict(config, **{
            "env": TwoStepGame,
        }),
    )


def test_maddpg_custom_metrics():
    def on_episode_start(info):
        pass
        # episode = info["episode"]
        # print("episode {} started".format(episode.episode_id))
        #
        # # episode.user_data["di"]
        #
        # # Add whatever you like here to serve for episode_step callback.
        # episode.user_data["pole_angles"] = []

    def on_episode_step(info):
        pass
        # episode = info["episode"]
        # pole_angle = abs(episode.last_observation_for()[2])
        # raw_angle = abs(episode.last_raw_obs_for()[2])
        # assert pole_angle == raw_angle
        # episode.user_data["pole_angles"].append(pole_angle)

    def on_episode_end(info):
        pass
        # episode = info["episode"]
        # print('episode')
        # pole_angle = np.mean(episode.user_data["pole_angles"])
        # print("episode {} ended with length {} and pole angles {}".format(
        #     episode.episode_id, episode.length, pole_angle))
        # episode.custom_metrics["pole_angle"] = pole_angle

    def on_sample_end(info):
        """The info here contain two things, the worker
        and the samples. The samples is a 'MultiAgentBatch'
        """
        pass
        # batch = info['samples']
        # joint_dataset = SampleBatch.concat_samples([
        #     i for i in batch.policy_batches.values()
        # ])
        # # The size of joint_dataset is:
        # #   sample_batch_size * num_agents
        # # maybe we should only concat the observation.
        #
        # # timer = TimerStat()
        #
        # # with timer:
        #     # replay
        # joint_obs = joint_dataset[joint_dataset.CUR_OBS]
        # ret = {}
        # for pid, policy in info['worker'].policy_map.items():
        #     act, _, infos = policy.compute_actions(joint_obs)
        #     ret[pid] = [act, infos]
        #     # now we have a mapping: policy_id to joint_dataset_replay in 'ret'
        #
        # # print("Take time: ", timer.mean)
        #
        # flatten = [tup[0] for tup in ret.values()] # flatten action array
        # apply_function = lambda x, y: np.linalg.norm(x - y)
        # sunhao_matrix = _build_matrix(flatten, apply_function)
        # print("SUNHAO MATRIX: ", sunhao_matrix)

        # print("returned sample batch of size {}".format(info["samples"].count))

    def on_train_result(info):
        # print(
        #     "trainer.train() result: {} -> {} episodes".format(
        #         info["trainer"], info["result"]["episodes_this_iter"]
        #     )
        # )
        # # you can mutate the result dict to add new fields to return
        # info["result"]["callback_ok"] = True
        ###########################

        config = info['trainer'].config
        sample_size = config.get("joint_dataset_sample_size")
        if sample_size is None:
            print("[WARNING]!!! You do not specify the "
                  "joint_dataset_sample_size!! We will use 200 instead.")
            sample_size = 200

        # replay_buffers is a dict map policy_id to ReplayBuffer object.
        trainer = info['trainer']
        replay_buffers = trainer.optimizer.replay_buffers

        joint_obs = []
        for policy_id, replay_buffer in replay_buffers.items():
            obs = replay_buffer.sample(sample_size)[0]
            joint_obs.append(obs)
        joint_obs = np.concatenate(joint_obs)

        ret = {}
        if hasattr(trainer.workers, "policy_map"):
            iters = trainer.workers.policy_map.items()
        else:
            iters = trainer.workers.local_worker().policy_map.items()
        for pid, policy in iters:
            act, _, infos = policy.compute_actions(joint_obs)
            ret[pid] = [act, infos]
            # now we have a mapping: policy_id to joint_dataset_replay in 'ret'

        flatten = [tup[0] for tup in ret.values()]  # flatten action array
        apply_function = lambda x, y: np.linalg.norm(x - y)
        dist_matrix = _build_matrix(flatten, apply_function)

        mask = np.logical_not(
            np.diag(np.ones(dist_matrix.shape[0])).astype(np.bool)
        )
        flatten_dist = dist_matrix[mask]

        info['result']['distance_mean'] = flatten_dist.mean()
        info['result']['distance_max'] = flatten_dist.max()
        info['result']['distance_min'] = flatten_dist.min()

        # print("SUNHAO_MATRIX: ", sunhao_matrix)





    def on_postprocess_traj(info):
        episode = info["episode"]
        batch = info["post_batch"]
        # print("postprocessed {} steps".format(batch.count))
        if "num_batches" not in episode.custom_metrics:
            episode.custom_metrics["num_batches"] = 0
        episode.custom_metrics["num_batches"] += 1

    extra_config = {
        "callbacks": {
            "on_episode_start": on_episode_start,
            "on_episode_step": on_episode_step,
            "on_episode_end": on_episode_end,
            "on_sample_end": on_sample_end,
            "on_train_result": on_train_result,
            "on_postprocess_traj": on_postprocess_traj,
        },
    }

    test_maddpg_basic(extra_config, local_mode=False)


if __name__ == "__main__":
    test_maddpg_custom_metrics()
