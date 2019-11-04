import numpy as np
from ray import tune

from toolbox import initialize_ray
from toolbox.distance import joint_dataset_distance
from toolbox.env import get_env_maker
from toolbox.additional_loss.multiagent_env_wrapper import MultiAgentEnvWrapper


def test_marl_individual_ppo(extra_config, local_mode=True, test_mode=True):
    num_gpus = 0
    exp_name = "test_marl_individual_ppo"
    env_name = "BipedalWalker-v2"
    num_iters = 50
    num_agents = 8

    initialize_ray(test_mode=test_mode, num_gpus=num_gpus,
                   local_mode=local_mode)

    tmp_env = get_env_maker(env_name)()

    default_policy = (
        None, tmp_env.observation_space, tmp_env.action_space, {}
    )

    policy_names = ["ppo_agent{}".format(i) for i in range(num_agents)]

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
        """info only contains trainer and result."""
        sample_size = info['trainer'].config.get("joint_dataset_sample_size")
        if sample_size is None:
            print("[WARNING]!!! You do not specify the "
                  "joint_dataset_sample_size!! We will use 200 instead.")
            sample_size = 200

        # replay_buffers is a dict map policy_id to ReplayBuffer object.
        trainer = info['trainer']
        worker = trainer.workers.local_worker()

        joint_obs = []
        if hasattr(trainer.optimizer, "replay_buffers"):
            # If we are using maddpg, it use ReplayOptimizer, which has this
            # attribute.
            for policy_id, replay_buffer in \
                    trainer.optimizer.replay_buffers.items():
                obs = replay_buffer.sample(sample_size)[0]
                joint_obs.append(obs)
        else:
            # If we are using individual PPO, it has no replay buffer,
            # so it seems we have to rollout here to collect the observations
            multi_agent_batch = worker.sample()

            for pid, batch in multi_agent_batch.policy_batches.items():
                count = batch.count
                batch.shuffle()
                if count < sample_size:
                    print("[WARNING]!!! Your rollout sample size is "
                          "less than the replay sample size! "
                          "Check codes here!")
                    cnt = 0
                    while True:
                        end = min(count, sample_size - cnt)
                        joint_obs.append(batch.slice(0, end)['obs'])
                        if end < count:
                            break
                        cnt += end
                        batch.shuffle()
                else:
                    joint_obs.append(batch.slice(0, sample_size)['obs'])

        joint_obs = np.concatenate(joint_obs)

        def _replay(policy, pid):
            act, _, infos = policy.compute_actions(joint_obs)
            return pid, act, infos

        ret = {
            pid: [act, infos]
            for pid, act, infos in worker.foreach_policy(_replay)
        }
        # now we have a mapping: policy_id to joint_dataset_replay in 'ret'

        flatten = [act for act, infos in ret.values()]  # flatten action array
        dist_matrix = joint_dataset_distance(flatten)

        mask = np.logical_not(
            np.diag(np.ones(dist_matrix.shape[0])).astype(np.bool)
        )
        flatten_dist = dist_matrix[mask]

        info['result']['distance'] = {}
        info['result']['distance']['overall_mean'] = flatten_dist.mean()
        info['result']['distance']['overall_max'] = flatten_dist.max()
        info['result']['distance']['overall_min'] = flatten_dist.min()

        for i, pid in enumerate(ret.keys()):
            row_without_self = dist_matrix[i][mask[i]]
            info['result']['distance'][pid + "_mean"] = row_without_self.mean()

    extra_config = {"callbacks": {
        "on_train_result": on_train_result
    }}

    test_marl_individual_ppo(extra_config, local_mode=True, test_mode=False)


if __name__ == '__main__':
    # test_marl_individual_ppo()
    test_marl_custom_metrics()
