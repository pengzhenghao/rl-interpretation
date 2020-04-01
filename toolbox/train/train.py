import argparse

from ray import tune

from toolbox.train.deprecated_train_config import get_config
from toolbox.utils import initialize_ray, get_local_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--env", type=str, default="BipedalWalker-v2")
    parser.add_argument("--run", type=str, default="PPO")
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--test-mode", action="store_true")
    args = parser.parse_args()

    print("Argument: ", args)

    run_config, algo_specify_config = get_config(args.env, args.run,
                                                 args.test_mode)

    initialize_ray(num_gpus=args.num_gpus, test_mode=args.test_mode)
    tune.run(
        args.run,
        name=args.exp_name,
        verbose=1,
        local_dir=get_local_dir(),
        checkpoint_freq=1,
        checkpoint_at_end=True,
        stop={"timesteps_total": algo_specify_config['timesteps_total']}
        if "timesteps_total" in algo_specify_config \
            else algo_specify_config['stop'],
        config=run_config,
    )
