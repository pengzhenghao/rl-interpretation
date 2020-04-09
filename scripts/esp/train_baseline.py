"""
This file only provide training for baseline
"""
import shutil
import tempfile
import time

from toolbox import initialize_ray
from toolbox.evolution import GaussianESTrainer
from toolbox.train import train, get_train_parser

if __name__ == '__main__':
    parser = get_train_parser()
    parser.add_argument("--ppo", action="store_true")
    parser.add_argument("--es", action="store_true")
    parser.add_argument("--es-large", action="store_true")
    parser.add_argument("--es-optimizer", type=str, default="sgd")
    parser.add_argument("--stop", type=float, default=1e7)
    parser.add_argument("--local-mode", "-lm", action="store_true")
    args = parser.parse_args()
    print(args)
    local_mode = args.local_mode
    env_name = "CartPole-v0"
    dir_path = tempfile.mkdtemp()
    now = time.time()
    num_gpus = 0
    initialize_ray(test_mode=False, local_mode=local_mode, num_gpus=1)
    assert int(args.ppo) + int(args.es) + int(args.es_large) == 1
    if args.ppo:
        config = {
            "env": env_name,
            "num_sgd_iter": 10,
            "num_gpus": num_gpus,
            "train_batch_size": 4000,
            "sample_batch_size": 200,
            "num_envs_per_worker": 8,
            "lr": 2.5e-4,
        }
    if args.es or args.es_large:
        config = {
            "train_batch_size": 4000,
            "num_workers": 10,
            "optimizer_type": args.es_optimizer, "env": env_name,
            "lr": 2.5e-4,
            "episodes_per_batch": 1,
            "num_cpus_per_worker": 0.5
        }
        run = GaussianESTrainer
        if args.es_large:
            config.update({
                "episodes_per_batch": 1000,
                "train_batch_size": 10000
            })
    train(
        run, stop=int(args.stop), verbose=2, extra_config=config,
        exp_name=args.exp_name, num_seeds=args.num_seeds, num_gpus=args.num_gpus
    )
    shutil.rmtree(dir_path, ignore_errors=True)
    print("Test finished! Cost time: ", time.time() - now)
