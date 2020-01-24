import argparse
import pickle

from ray import tune

from toolbox import initialize_ray, get_local_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--env-name", type=str, default="Walker2d-v3")
    parser.add_argument("--address", type=str, default="")
    args = parser.parse_args()

    exp_name = args.exp_name  # It's "12230-ppo..." previously....
    env_name = args.env_name

    is_humanoid = "Humanoid" in env_name
    assert is_humanoid

    stop = int(5e8)
    num_gpus = args.num_gpus

    walker_config = {
        "seed": tune.grid_search([i * 100 for i in range(3)]),
        "env": env_name,

        # should be fixed
        "kl_coeff": 1.0,
        "gamma": 0.995,
        "lambda": 0.95,
        "clip_param": 0.2,
        "num_sgd_iter": 20,
        "lr": 0.0001,
        "horizon": 5000,

        'sample_batch_size': 200,
        'sgd_minibatch_size': 10000,
        'train_batch_size': 100000,
        "num_gpus": 1,
        "num_cpus_per_worker": 1,
        "num_cpus_for_driver": 1,
        "num_envs_per_worker": 16,
        'num_workers': 16,
    }

    initialize_ray(
        test_mode=False,
        local_mode=False,
        num_gpus=num_gpus if not args.address else None,
        address=args.address if args.address else None
    )

    if "Bullet" in env_name:
        from ray.tune.registry import register_env


        def make_pybullet(_=None):
            import pybullet_envs
            import gym
            print("Successfully import pybullet and found: ",
                  pybullet_envs.getList())
            return gym.make(env_name)


        register_env(env_name, make_pybullet)

    analysis = tune.run(
        "PPO",
        local_dir=get_local_dir(),
        name=exp_name,
        checkpoint_freq=10,
        keep_checkpoints_num=10,
        checkpoint_score_attr="episode_reward_mean",
        checkpoint_at_end=True,
        stop={"info/num_steps_sampled": stop}
        if isinstance(stop, int) else stop,
        config=walker_config,
        max_failures=20,
        reuse_actors=False
    )

    path = "{}-{}-{}ts.pkl".format(
        exp_name, env_name, stop
    )

    with open(path, "wb") as f:
        data = analysis.fetch_trial_dataframes()
        pickle.dump(data, f)
        print("Result is saved at: <{}>".format(path))
