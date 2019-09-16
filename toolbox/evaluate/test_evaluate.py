import gym
import ray
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy

from toolbox.evaluate.rollout import RolloutWorkerWrapper, \
    several_agent_rollout, rollout, make_worker, efficient_rollout_from_worker
from toolbox.env.env_maker import get_env_maker
from toolbox.utils import initialize_ray


def test_RolloutWorkerWrapper_with_activation():
    initialize_ray(test_mode=True)
    env_maker = lambda _: gym.make("BipedalWalker-v2")
    ckpt = "test/fake-ckpt1/checkpoint-313"
    rww_new = RolloutWorkerWrapper.as_remote().remote(True)
    rww_new.reset.remote(
        ckpt=ckpt, num_rollouts=2, seed=0,
        env_creater=env_maker, run_name="PPO", env_name="BipedalWalker-v2",
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
    worker = make_worker(get_env_maker(env_name),
                            None, 1, 0, "ES", env_name)
    trajctory_list = efficient_rollout_from_worker(worker)
    print(trajctory_list)
    return trajctory_list


if __name__ == '__main__':
    r = test_efficient_rollout_from_worker()
