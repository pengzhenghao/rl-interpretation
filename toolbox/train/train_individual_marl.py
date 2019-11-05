import argparse

import numpy as np
from ray import tune

from toolbox.distance import joint_dataset_distance, js_distance
from toolbox.env import get_env_maker
from toolbox.marl import MultiAgentEnvWrapper
from toolbox.utils import get_num_gpus, get_local_dir, initialize_ray

from toolbox.marl.extra_loss_ppo_trainer import ExtraLossPPOTrainer


def _collect_joint_dataset(trainer, worker, sample_size):
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
    return joint_obs


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
    joint_obs = _collect_joint_dataset(trainer, worker, sample_size)

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

    js_matrix = js_distance(flatten)
    flatten_js_dist = js_matrix[mask]
    info['result']['distance_js'] = {}
    info['result']['distance_js']['overall_mean'] = flatten_js_dist.mean()
    info['result']['distance_js']['overall_max'] = flatten_js_dist.max()
    info['result']['distance_js']['overall_min'] = flatten_js_dist.min()
    for i, pid in enumerate(ret.keys()):
        row_without_self = js_matrix[i][mask[i]]
        info['result']['distance_js'][pid + "_mean"] = row_without_self.mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--env", type=str, default="BipedalWalker-v2")
    parser.add_argument("--run", type=str, default="PPO")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--num-agents", type=int, default=10)
    parser.add_argument("--num-seeds", type=int, default=0)
    parser.add_argument("--num-timesteps", type=float, default=5e6)
    parser.add_argument("--test-mode", action="store_true")
    args = parser.parse_args()

    exp_name = args.exp_name
    env_name = args.env
    num_timesteps = int(args.num_timesteps)
    run_name = args.run

    run_object = {
        "individual": "PPO",
        "extra_loss": ExtraLossPPOTrainer
    }[run_name]

    initialize_ray(num_gpus=args.num_gpus, test_mode=args.test_mode,
                   object_store_memory=40 * 1024 * 1024 * 1024)

    policy_names = ["ppo_agent{}".format(i) for i in range(args.num_agents)]

    tmp_env = get_env_maker(env_name)()
    default_policy = (
        None, tmp_env.observation_space, tmp_env.action_space, {}
    )

    config = {
        "env": MultiAgentEnvWrapper,
        "env_config": {
            "env_name": env_name,
            "agent_ids": policy_names
        },
        "log_level": "DEBUG" if args.test_mode else "ERROR",
        "num_gpus": get_num_gpus(args.num_seeds + 1),
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
            list(range(args.num_seeds))) if args.num_seeds != 0 else 0
    }

    tune.run(
        run_object,
        local_dir=get_local_dir(),
        name=exp_name,
        checkpoint_at_end=True,
        checkpoint_freq=10,
        stop={"timesteps_total": num_timesteps},
        config=config,
    )
