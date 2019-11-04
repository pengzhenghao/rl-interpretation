import numpy as np
from ray import tune

from toolbox import initialize_ray
from toolbox.env import get_env_maker
from toolbox.marl.multiagent_env_wrapper import MultiAgentEnvWrapper


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


def test_marl_individual_ppo(extra_config, local_mode=True):
    num_gpus = 4
    exp_name = "test_marl_individual_ppo"
    env_name = "BipedalWalker-v2"
    num_iters = 50
    num_agents = 4

    initialize_ray(test_mode=True, num_gpus=num_gpus, local_mode=local_mode)

    tmp_env = get_env_maker(env_name)()

    default_policy = (
        None, tmp_env.observation_space, tmp_env.action_space, {}
    )

    policy_names = ["Agent{}".format(i) for i in range(num_agents)]

    def policy_mapping_fn(aid):
        # print("input aid: ", aid)
        return aid

    config = {
        "env": MultiAgentEnvWrapper,
        "env_config": {
            "env_name": env_name,
            "agent_ids": policy_names
        },
        "log_level": "DEBUG",
        "num_gpus": num_gpus,
        "multiagent": {
            "policies": {i: default_policy
                         for i in policy_names},
            "policy_mapping_fn": policy_mapping_fn,
        },
    }

    if isinstance(extra_config, dict):
        config.update(extra_config)

    tune.run(
        "PPO",
        name=exp_name,
        checkpoint_at_end=True,
        checkpoint_freq=10,
        stop={"training_iteration": num_iters},
        config=config,
    )


def test_marl_custom_metrics():
    def on_train_result(info):
        sample_size = info['trainer'].config.get("joint_dataset_sample_size")
        if sample_size is None:
            print("[WARNING]!!! You do not specify the "
                  "joint_dataset_sample_size!! We will use 200 instead.")
            sample_size = 200

        # replay_buffers is a dict map policy_id to ReplayBuffer object.
        trainer = info['trainer']

        joint_obs = []
        for policy_id, replay_buffer in \
                trainer.optimizer.replay_buffers.items():
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

    extra_config = {"callbacks": {
        "on_train_result": on_train_result
    }}

    test_marl_individual_ppo(extra_config, local_mode=False)


if __name__ == '__main__':
    # test_marl_individual_ppo()
    test_marl_custom_metrics()
