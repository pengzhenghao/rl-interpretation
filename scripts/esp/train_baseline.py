"""
This file only provide training for baseline
"""
import time

from ray import tune
from ray.rllib.agents.ppo.ppo import PPOTrainer, PPOTFPolicy

from toolbox import train
from toolbox.evolution import GaussianESTrainer
from toolbox.evolution_plugin.evolution_plugin import choose_optimzier, \
    merge_dicts, DEFAULT_CONFIG
from toolbox.train import get_train_parser

ppo_sgd_config = merge_dicts(DEFAULT_CONFIG, dict(master_optimizer_type="sgd"))

PPOSGDPolicy = PPOTFPolicy.with_updates(
    name="PPOSGDPolicy",
    get_default_config=lambda: ppo_sgd_config,
    optimizer_fn=choose_optimzier
)

PPOSGDTrainer = PPOTrainer.with_updates(
    name="PPOSGD",
    default_config=ppo_sgd_config,
    default_policy=PPOSGDPolicy,
    get_policy_class=lambda _: PPOSGDPolicy
)

if __name__ == '__main__':
    parser = get_train_parser()
    parser.add_argument("--ppo", action="store_true")
    parser.add_argument("--es", action="store_true")
    # parser.add_argument("--optimizer", type=str, default="sgd")  # [adam, sgd]
    parser.add_argument("--stop", type=float, default=1e7)
    parser.add_argument("--local-mode", "-lm", action="store_true")
    args = parser.parse_args()
    print(args)
    local_mode = args.local_mode
    now = time.time()
    assert int(args.ppo) + int(args.es) == 1

    if args.ppo:
        run = PPOSGDTrainer
        config = {
            # config = {
            "env": tune.grid_search([
                'Walker2DBulletEnv-v0',
                'HalfCheetahBulletEnv-v0',
                'AntBulletEnv-v0',
                'HopperBulletEnv-v0',
            ]),

            "num_sgd_iter": 10,
            "train_batch_size": 4000,
            "sample_batch_size": 200,
            "num_envs_per_worker": 8,
            "entropy_coeff": 0.001,
            "lambda": 0.95,
            "lr": 2.5e-4,
            "num_gpus": 0.5,  # 16 trials in one machine

            # locomotion config
            "kl_coeff": 1.0,

            # SGD optimizer
            "master_optimizer_type": tune.grid_search(["sgd", "adam"])
        }
        stop = int(1e7)

    if args.es:
        config = {
            "env": tune.grid_search([
                'Walker2DBulletEnv-v0',
                'HalfCheetahBulletEnv-v0',
                'AntBulletEnv-v0',
                'HopperBulletEnv-v0',
            ]),

            "train_batch_size": 4000,
            "num_workers": 10,
            "optimizer_type": tune.grid_search(["sgd", "adam"]),
            "num_gpus": 0,
            "lr": 2.5e-4,
            "episodes_per_batch": 1,
            "num_cpus_per_worker": 0.5
        }
        run = GaussianESTrainer
        stop = int(1e8)

    train(
        run, stop=int(args.stop), verbose=1, config=config,
        exp_name=args.exp_name, num_seeds=args.num_seeds, num_gpus=args.num_gpus
    )

    print("Test finished! Cost time: ", time.time() - now)
