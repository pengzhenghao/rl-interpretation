import numpy as np
from ray import tune

from toolbox import initialize_ray
from toolbox.env import get_env_maker
from toolbox.marl.multiagent_env_wrapper import MultiAgentEnvWrapper


def test_marl_individual_ppo():
    num_gpus = 4
    exp_name = "test_marl_individual_ppo"
    env_name = "BipedalWalker-v2"
    num_iters = 50
    num_agents = 8

    initialize_ray(test_mode=True, num_gpus=num_gpus)

    tmp_env = get_env_maker(env_name)()

    default_policy = (
        None,
        tmp_env.observation_space,
        tmp_env.action_space,
        {}
    )

    policy_names = ["Agent{}".format(i) for i in range(num_agents)]

    def policy_mapping_fn(aid):
        print("input aid: ", aid)
        return aid

    tune.run(
        "PPO",
        name=exp_name,
        checkpoint_at_end=True,
        checkpoint_freq=10,
        stop={"training_iteration": num_iters},
        config={
            "env": MultiAgentEnvWrapper,
            "env_config": {
                "env_name": env_name,
                "agent_ids": policy_names
            },
            "log_level": "DEBUG",
            "num_gpus": num_gpus,
            "multiagent": {
                "policies": {
                    i: default_policy for i in policy_names
                },
                "policy_mapping_fn": policy_mapping_fn,
            },
        },
    )


def test_marl_custom_metrics():
    num_gpus = 0
    exp_name = "test_marl_individual_ppo"
    env_name = "BipedalWalker-v2"
    num_iters = 20
    num_agents = 2

    def on_episode_start(info):
        episode = info["episode"]
        print("episode {} started".format(episode.episode_id))

        # episode.user_data["di"]

        # Add whatever you like here to serve for episode_step callback.
        episode.user_data["pole_angles"] = []

    def on_episode_step(info):
        pass
        # episode = info["episode"]
        # pole_angle = abs(episode.last_observation_for()[2])
        # raw_angle = abs(episode.last_raw_obs_for()[2])
        # assert pole_angle == raw_angle
        # episode.user_data["pole_angles"].append(pole_angle)

    def on_episode_end(info):
        episode = info["episode"]
        print('episode')
        # pole_angle = np.mean(episode.user_data["pole_angles"])
        # print("episode {} ended with length {} and pole angles {}".format(
        #     episode.episode_id, episode.length, pole_angle))
        # episode.custom_metrics["pole_angle"] = pole_angle

    def on_sample_end(info):
        print("returned sample batch of size {}".format(info["samples"].count))

    def on_train_result(info):
        print("trainer.train() result: {} -> {} episodes".format(
            info["trainer"], info["result"]["episodes_this_iter"]))
        # you can mutate the result dict to add new fields to return
        info["result"]["callback_ok"] = True

    def on_postprocess_traj(info):
        episode = info["episode"]
        batch = info["post_batch"]
        print("postprocessed {} steps".format(batch.count))
        if "num_batches" not in episode.custom_metrics:
            episode.custom_metrics["num_batches"] = 0
        episode.custom_metrics["num_batches"] += 1

    initialize_ray(test_mode=True, num_gpus=num_gpus, local_mode=True)

    tmp_env = get_env_maker(env_name)()

    default_policy = (
        None,
        tmp_env.observation_space,
        tmp_env.action_space,
        {}
    )

    policy_names = ["Agent{}".format(i) for i in range(num_agents)]

    def policy_mapping_fn(aid):
        print("input aid: ", aid)
        return aid

    tune.run(
        "PPO",
        name=exp_name,
        checkpoint_at_end=True,
        checkpoint_freq=10,
        stop={"training_iteration": num_iters},
        config={
            "env": MultiAgentEnvWrapper,
            "env_config": {
                "env_name": env_name,
                "agent_ids": policy_names
            },
            "callbacks": {
                "on_episode_start": on_episode_start,
                "on_episode_step": on_episode_step,
                "on_episode_end": on_episode_end,
                "on_sample_end": on_sample_end,
                "on_train_result": on_train_result,
                "on_postprocess_traj": on_postprocess_traj,
            },
            "log_level": "DEBUG",
            "num_gpus": num_gpus,
            "multiagent": {
                "policies": {
                    i: default_policy for i in policy_names
                },
                "policy_mapping_fn": policy_mapping_fn,
            },
        },
    )


if __name__ == '__main__':
    # test_marl_individual_ppo()
    test_marl_custom_metrics()
