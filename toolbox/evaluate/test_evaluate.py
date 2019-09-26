import gym
import numpy as np
import ray
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy

from toolbox.env.env_maker import get_env_maker
from toolbox.evaluate.evaluate_utils import restore_agent_with_mask, \
    restore_agent_with_activation, restore_agent
from toolbox.evaluate.rollout import RolloutWorkerWrapper, \
    several_agent_rollout, rollout, make_worker, efficient_rollout_from_worker
from toolbox.utils import initialize_ray


def test_RolloutWorkerWrapper_with_activation():
    initialize_ray(test_mode=True)
    env_maker = lambda _: gym.make("BipedalWalker-v2")
    ckpt = "test/fake-ckpt1/checkpoint-313"
    rww_new = RolloutWorkerWrapper.as_remote().remote(True)
    rww_new.reset.remote(
        ckpt=ckpt,
        num_rollouts=2,
        seed=0,
        env_creater=env_maker,
        run_name="PPO",
        env_name="BipedalWalker-v2",
        require_activation=True
    )
    for _ in range(2):
        result = ray.get(rww_new.wrap_sample.remote())
    print(result)
    print("Prepare to close")
    print("Dataset: ", ray.get(rww_new.get_dataset.remote()))
    rww_new.close.remote()
    print("After close")
    return result


def test_serveral_agent_rollout(force=False):
    yaml_path = "data/0811-random-test.yaml"
    num_rollouts = 2
    initialize_ray()
    return several_agent_rollout(
        yaml_path, num_rollouts, force_rewrite=force, return_data=True
    )


def _test_es_agent_compatibility():
    from ray.rllib.agents.es import ESTrainer
    es = ESTrainer(env="BipedalWalker-v2")
    env = gym.make("BipedalWalker-v2")
    rollout(es, env, "BipedalWalker-v2", num_steps=100, require_frame=True)


def test_RolloutWorkerWrapper():
    initialize_ray(test_mode=True)
    env_maker = lambda _: gym.make("BipedalWalker-v2")
    ckpt = "test/fake-ckpt1/checkpoint-313"
    # rww = RolloutWorkerWrapper(ckpt, 2, 0, env_maker, PPOTFPolicy)
    # for _ in range(2):
    #     result = rww.wrap_sample()
    # print(result)
    # rww.close()

    rww_new = RolloutWorkerWrapper.as_remote().remote(
        ckpt,
        2,
        0,
        env_maker,
        PPOTFPolicy,
        run_name="PPO",
        env_name="BipedalWalker-v2"
    )
    for _ in range(2):
        result = ray.get(rww_new.wrap_sample.remote())
    print(result)
    print("Prepare to close")
    print("Dataset: ", ray.get(rww_new.get_dataset.remote()))
    rww_new.close.remote()
    print("After close")


def test_efficient_rollout_from_worker():
    initialize_ray(test_mode=True)
    env_name = "BipedalWalker-v2"
    worker = make_worker(get_env_maker(env_name), None, 1, 0, "ES", env_name)
    trajctory_list = efficient_rollout_from_worker(worker)
    print(trajctory_list)
    return trajctory_list


def test_restore_agent_with_mask():
    initialize_ray(test_mode=True)
    env_name = "BipedalWalker-v2"
    ckpt = "~/ray_results/0810-20seeds/PPO_BipedalWalker-v2_0_seed=20_" \
           "2019-08-10_16-54-37xaa2muqm/checkpoint_469/checkpoint-469"
    agent = restore_agent_with_mask("PPO", ckpt, env_name)

    obs_space = agent.get_policy().observation_space
    obs = obs_space.sample()

    val = np.zeros((1, 256), dtype=np.float32)
    val[:, ::2] = 1.0

    mask_batch = {"fc_1_mask": val.copy(), "fc_2_mask": val.copy()}

    ret = agent.get_policy().compute_actions(
        np.array([obs]), mask_batch=mask_batch
    )

    return ret


def test_rollout():
    initialize_ray(test_mode=True, num_gpus=1)
    env_name = "HalfCheetah-v2"
    a = restore_agent("PPO", None, env_name)
    env = gym.make(env_name)

    obs = env.observation_space.sample()

    print(obs.shape)

    policy = a.get_policy()

    ret = policy.compute_single_action(obs, [])

    return ret


if __name__ == '__main__':
    ret = test_rollout()
    print("ret of normal rollout: ", ret)
    ret = test_restore_agent_with_mask()
