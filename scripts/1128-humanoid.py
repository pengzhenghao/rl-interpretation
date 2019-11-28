import pybullet_envs
from ray import tune

from toolbox.env import get_env_maker
from toolbox.marl import MultiAgentEnvWrapper, on_train_result
from toolbox.marl.smart_adaptive_extra_loss import \
    SmartAdaptiveExtraLossPPOTrainer
from toolbox.utils import get_local_dir, initialize_ray


if pybullet_envs:
    print("Successfully import pybullet_envs!")

if __name__ == '__main__':
    num_gpus = 4
    num_agents = 10
    env_name = "HumanoidBulletEnv-v0"
    num_seeds = 1
    exp_name = "1128-SAEL-humanoid"

    initialize_ray(
        num_gpus=num_gpus,
        object_store_memory=40 * 1024 * 1024 * 1024
    )
    policy_names = ["ppo_agent{}".format(i) for i in range(num_agents)]
    tmp_env = get_env_maker(env_name)()
    default_policy = (
        None, tmp_env.observation_space, tmp_env.action_space, {}
    )

    config = {
        # This experiment specify config
        # "performance_evaluation_metric": tune.grid_search(['max', 'mean']),
        "use_joint_dataset": tune.grid_search([True, False]),
        "novelty_mode": tune.grid_search(['min', 'mean']),

        # Humanoid Specify parameter
        "gamma": 0.995,
        "lambda": 0.95,
        "clip_param": 0.2,
        "kl_coeff": 1.0,
        "lr": 0.0003,
        "horizon": 5000,
        'sgd_minibatch_size': 4096,
        'train_batch_size': 16384,
        "num_workers": 16,
        "num_gpus": 1,

        # some common config copied from train_individual_marl
        "env": MultiAgentEnvWrapper,
        "env_config": {
            "env_name": env_name,
            "agent_ids": policy_names
        },
        "log_level": "ERROR",
        "num_cpus_per_worker": 2,
        "num_cpus_for_driver": 1,
        "num_envs_per_worker": 16,
        "sample_batch_size": 256,
        "multiagent": {
            "policies": {i: default_policy
                         for i in policy_names},
            "policy_mapping_fn": lambda aid: aid,
        },
        "callbacks": {
            "on_train_result": on_train_result
        },
        "num_sgd_iter": 10,
        "seed": tune.grid_search(
            list(range(num_seeds))) if num_seeds > 1 else 0
    }

    tune.run(
        SmartAdaptiveExtraLossPPOTrainer,
        local_dir=get_local_dir(),
        name=exp_name,
        checkpoint_at_end=True,
        checkpoint_freq=10,
        stop={
            "timesteps_total": int(1e8),
            "episode_reward_mean": 310 * num_agents
        },
        config=config,
    )
