import gym
import numpy as np
import ray
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy

from toolbox.env.env_maker import get_env_maker
from toolbox.evaluate.evaluate_utils import restore_agent_with_mask, restore_policy_with_mask
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


# from toolbox.evaluate.symbolic_agent import add_gaussian_perturbation
import copy


def get_policy_network_output(agent, obs):
    act, _, info = agent.get_policy().compute_actions(obs)
    print(info.keys())
    return info['behaviour_logits']


def test_add_gaussian_perturbation():
    initialize_ray(test_mode=True)
    agent = restore_agent_with_mask("PPO", None, "BipedalWalker-v2")

    act = np.ones((1, 24))

    old_response = copy.deepcopy(get_policy_network_output(agent, act))
    old_response2 = copy.deepcopy(get_policy_network_output(agent, act))
    agent = MaskSymbolicAgent.add_gaussian_perturbation(agent, 1.0, 0.0, 1997)
    new_response = copy.deepcopy(get_policy_network_output(agent, act))

    np.testing.assert_array_equal(old_response, old_response2)
    np.testing.assert_array_equal(old_response, new_response)

    agent2 = MaskSymbolicAgent.add_gaussian_perturbation(agent, 1.0, 1, 1997)
    new_response2 = copy.deepcopy(get_policy_network_output(agent2, act))

    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal, old_response,
        new_response2
    )


from toolbox.evaluate.symbolic_agent import MaskSymbolicAgent
from toolbox.process_data.process_data import read_yaml


def test_MaskSymbolicAgent_local():
    initialize_ray(test_mode=True)

    name_ckpt_mapping = read_yaml("../../data/yaml/test-2-agents.yaml")

    ckpt_info = next(iter(name_ckpt_mapping.values()))

    sa = MaskSymbolicAgent(ckpt_info)

    agent = sa.get()['policy']

    callback_info = {"method": 'normal', 'mean': 1., "std": 1., "seed": 1997}

    sa2 = MaskSymbolicAgent(ckpt_info, callback_info)

    agent2 = sa2.get()['policy']


def test_MaskSymbolicAgent_remote():
    initialize_ray(test_mode=True)

    # @ray.remote
    # def get_agent(sa):
    #     agent = sa.get()['agent']
    #     print("success")
    #     return True

    @ray.remote
    def get_policy(sa):
        policy = sa.get()['policy']
        print("success")
        return True

    name_ckpt_mapping = read_yaml("../../data/yaml/test-2-agents.yaml")

    # ckpt_info = next(iter(name_ckpt_mapping.values()))

    callback_info = {"method": 'normal', 'mean': 1., "std": 1., "seed": 1997}
    obidlist = []

    for name, ckpt_info in name_ckpt_mapping.items():
        sa = MaskSymbolicAgent(ckpt_info, callback_info)
        obid = get_policy.remote(sa)
        obidlist.append(obid)

    ray.get(obidlist)


def test_restore_agent_and_restore_policy():

    initialize_ray(test_mode=True)

    name_ckpt_mapping = read_yaml("../../data/yaml/test-2-agents.yaml")
    ckpt_info = next(iter(name_ckpt_mapping.values()))

    pure_agent = restore_agent_with_mask(
        "PPO", ckpt_info['path'], "BipedalWalker-v2"
    )

    policy = restore_policy_with_mask(
        "PPO", ckpt_info['path'], "BipedalWalker-v2"
    )

    np.testing.assert_almost_equal(
        pure_agent.get_policy().get_state(), policy.get_state()
    )

    for i in range(10):
        a = np.random.random((10, 24))
        pr, _, infopr = policy.compute_actions(a)
        ar, _, infoar = pure_agent.get_policy().compute_actions(a)
        #     print(infopr.keys())
        np.testing.assert_almost_equal(
            infopr['behaviour_logits'], infoar['behaviour_logits']
        )


if __name__ == '__main__':
    r = test_efficient_rollout_from_worker()
    ret = test_restore_agent_with_mask()
    test_add_gaussian_perturbation()
    test_MaskSymbolicAgent_local()
    test_MaskSymbolicAgent_remote()
    test_restore_agent_and_restore_policy()
