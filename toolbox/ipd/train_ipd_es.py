import argparse

import ray

from toolbox import initialize_ray
from toolbox.ipd.ipd import IPDEnv, IPDTrainer
from toolbox.ipd.train_tnb import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="")
    parser.add_argument("--env-name", type=str, required=True)
    parser.add_argument("--address", type=str, default="")

    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--test-mode", action="store_true")
    args = parser.parse_args()

    if not args.test_mode:
        assert args.exp_name

    common_config = {
        "env": IPDEnv,
        "env_config": {
            "env_name": args.env_name
        },
        "kl_coeff": 1.0,
        "num_sgd_iter": 20,
        "lr": 0.0002,
        'sample_batch_size': 256,
        'sgd_minibatch_size': 4096,
        'train_batch_size': 65536,
        "num_gpus": 0.4,
        "num_cpus_per_worker": 0.4,
        "num_cpus_for_driver": 0.4,
        "num_envs_per_worker": 16,
        'num_workers': 8,
    }

    config = common_config
    if args.env_name == "Walker2d-v3":
        timesteps = int(2e7)
        config['novelty_threshold'] = 1.1
        config['env_config']['novelty_threshold'] = 1.1
    elif args.env_name == "Hopper-v3":
        timesteps = int(2e7)
        config['env_config']['novelty_threshold'] = 0.6
    elif args.env_name == "HalfCheetah-v3":
        timesteps = int(5e7)
        config.update(
            {
                "gamma": 0.99,
                "lambda": 0.95,
                "kl_coeff": 1.0,
                'num_sgd_iter': 32,
                'lr': 0.0003,
                'vf_loss_coeff': 0.5,
                'clip_param': 0.2,
                "grad_clip": 0.5,
                'novelty_threshold': 1.3
            }
        )
        config['env_config']['novelty_threshold'] = 1.3
    else:
        raise NotImplementedError()

    def ray_init():
        ray.shutdown()
        initialize_ray(
            test_mode=args.test_mode,
            local_mode=False,
            num_gpus=args.num_gpus if not args.address else None,
            redis_address=args.address if args.address else None
        )

    main(
        trainer=IPDTrainer,
        exp_name=args.exp_name,
        num_iterations=100,
        max_num_agents=10,
        timesteps_total=int(timesteps),
        common_config=common_config,
        max_not_improve_iterations=3,
        ray_init=ray_init,
        test_mode=args.test_mode,
        disable_early_stop=True
    )
