"""
This file only provide training for baseline
"""
import shutil
import tempfile
import time

from ray.rllib.agents.ppo.ppo import PPOTrainer, PPOTFPolicy

from toolbox import initialize_ray, train
from toolbox.evolution import GaussianESTrainer
from toolbox.evolution_plugin.evolution_plugin import choose_optimzier, \
    merge_dicts, DEFAULT_CONFIG
from toolbox.train import get_train_parser

ppo_sgd_config = merge_dicts(DEFAULT_CONFIG, dict(master_optimizer_type="sgd"))

PPOSGDPolicy = PPOTFPolicy.with_updates(
    name="EvolutionPluginTFPolicy",
    get_default_config=lambda: ppo_sgd_config,
    optimizer_fn=choose_optimzier
)

PPOSGDTrainer = PPOTrainer.with_updates(
    name="EvolutionPlugin",
    default_config=ppo_sgd_config,
    default_policy=PPOSGDPolicy,
    get_policy_class=lambda _: PPOSGDPolicy
)

if __name__ == '__main__':
    parser = get_train_parser()
    parser.add_argument("--ppo", action="store_true")
    parser.add_argument("--es", action="store_true")
    parser.add_argument("--es-large", action="store_true")
    parser.add_argument("--optimizer", type=str, default="sgd")  # [adam, sgd]
    parser.add_argument("--stop", type=float, default=1e7)
    parser.add_argument("--local-mode", "-lm", action="store_true")
    args = parser.parse_args()
    print(args)
    local_mode = args.local_mode
    now = time.time()
    assert int(args.ppo) + int(args.es) + int(args.es_large) == 1
    if args.ppo:
        run = PPOSGDTrainer
        config = {
            "env": args.env_name,
            "num_sgd_iter": 10,
            "num_gpus": 1 if args.num_gpus > 0 else 0,
            "train_batch_size": 4000,
            "sample_batch_size": 200,
            "num_envs_per_worker": 8,
            "lr": 2.5e-4,
            "master_optimizer_type": args.optimizer
        }
    if args.es or args.es_large:
        config = {
            "train_batch_size": 4000,
            "num_workers": 10,
            "optimizer_type": args.optimizer,
            "env": args.env_name,
            "lr": 2.5e-4,
            "episodes_per_batch": 1,
            "num_cpus_per_worker": 0.5
        }
        run = GaussianESTrainer
        if args.es_large:
            config.update({
                "episodes_per_batch": 1,
                "train_batch_size": 10000
            })
    train(
        run, stop=int(args.stop), verbose=2, extra_config=config,
        exp_name=args.exp_name, num_seeds=args.num_seeds, num_gpus=args.num_gpus
    )
    print("Test finished! Cost time: ", time.time() - now)
