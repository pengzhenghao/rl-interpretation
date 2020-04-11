import argparse
import pickle

from ray import tune

from toolbox import initialize_ray, get_local_dir
from toolbox import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--env-name", type=str, default="Walker2d-v3")
    parser.add_argument("--address", type=str, default="")
    args = parser.parse_args()

    exp_name = args.exp_name  # It's "12230-ppo..." previously....
    env_name = args.env_name
    stop = int(5e6)
    num_gpus = args.num_gpus

    walker_config = {
        "seed": tune.grid_search([i * 100 for i in range(3)]),
        "env": env_name,
        # should be fixed
        "kl_coeff": 1.0,
        "num_sgd_iter": 10,
        "lr": 0.0002,
        'sample_batch_size': 50,
        'sgd_minibatch_size': 64,
        'train_batch_size': 2048,
        "num_gpus": 0.25,
        "num_cpus_per_worker": 1,
        "num_cpus_for_driver": 1,
        "num_envs_per_worker": 5,
        'num_workers': 1,
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

    train(
        "SAC",
        exp_name=exp_name,

    )

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
