import argparse
import pickle

from ray import tune

from toolbox import initialize_ray, get_local_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="0206-a2c-baseline")
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--env-name", type=str, default="Walker2d-v3")
    parser.add_argument("--address", type=str, default="")
    parser.add_argument("--stop", type=float, default=5e6)
    args = parser.parse_args()

    exp_name = "{}-{}".format(args.exp_name, args.env_name)
    env_name = args.env_name
    stop = int(args.stop)
    num_gpus = args.num_gpus

    walker_config = {
        "seed": tune.grid_search([i * 100 for i in range(3)]),
        "env": env_name,
        # should be fixed
        "lr": 0.0001,
        'train_batch_size': 2000,
        "num_gpus": 0.4,
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

    analysis = tune.run(
        "A2C",
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
        reuse_actors=False,
        verbose=1
    )

    path = "{}-{}-{}ts.pkl".format(
        exp_name, env_name, stop
    )

    with open(path, "wb") as f:
        data = analysis.fetch_trial_dataframes()
        pickle.dump(data, f)
        print("Result is saved at: <{}>".format(path))
