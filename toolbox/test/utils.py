import os

from toolbox.evaluate.rollout import RolloutWorkerWrapper
from toolbox.utils import build_env


def get_test_agent_config():
    return {
        "env_maker":
        build_env,
        "ckpt":
        os.path.abspath(
            os.path.expanduser(
                "~/ray_results/0810-20seeds/"
                "PPO_BipedalWalker-v2_0_seed=0_2019-08-10_15-21-164grca382/"
                "checkpoint_313/checkpoint-313"
            )
        ),
        "env_name":
        "BipedalWalker-v2",
        "run_name":
        "PPO"
    }


def load_test_agent_rollout():
    config = get_test_agent_config()
    num_rollout = 2
    seed = 0
    rww_new = RolloutWorkerWrapper()
    rww_new.reset(
        ckpt=config['ckpt'],
        num_rollouts=num_rollout,
        seed=seed,
        env_creater=config['env_maker'],
        run_name=config['run_name'],
        env_name=config['env_name']
    )
    return rww_new.wrap_sample()


def get_ppo_agent(env="CartPole-v0"):
    from ray.rllib.agents.ppo import PPOAgent
    return PPOAgent(env=env)

# def get_mujoco_agent(env=""):