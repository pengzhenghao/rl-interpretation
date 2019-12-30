import pickle

from ray import tune

from toolbox import initialize_ray, get_local_dir

if __name__ == '__main__':
    exp_name = "12230-ppo-pure-baseline"
    env_name = "Walker2d-v3"
    stop = int(5e7)
    num_gpus = 4

    walker_config = {
        "seed": tune.grid_search([i * 100 for i in range(3)]),
        "env": env_name,
        # should be fixed
        "kl_coeff": 1.0,
        "num_sgd_iter": 20,
        "lr": 0.0001,
        'sample_batch_size': 256,
        'sgd_minibatch_size': 4096,
        'train_batch_size': 65536,
        "num_gpus": 0.45,
        "num_cpus_per_worker": 0.5,
        "num_cpus_for_driver": 0.5,
        "num_envs_per_worker": 16,
        'num_workers': 8,
    }

    initialize_ray(
        test_mode=False,
        local_mode=False,
        num_gpus=num_gpus
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
