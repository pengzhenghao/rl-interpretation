import argparse
import shutil
import tempfile
import time

from ray import tune

from toolbox import initialize_ray
from toolbox.evolution_plugin import EPTrainer, HARD_FUSE, SOFT_FUSE

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--soft", action="store_true")  # default hard
    parser.add_argument("--ppo", action="store_true")
    parser.add_argument("--es", action="store_true")
    parser.add_argument("--es-optimizer", type=str, default="adam")
    parser.add_argument("--local-mode", "-lm", action="store_true")
    args = parser.parse_args()

    print(args)

    local_mode = args.local_mode
    env_name = "CartPole-v0"
    dir_path = tempfile.mkdtemp()
    now = time.time()
    num_gpus = 0

    initialize_ray(test_mode=True, local_mode=local_mode, num_gpus=1)

    config = {
        "env": env_name,
        "num_sgd_iter": 10,
        "num_gpus": num_gpus,
        "train_batch_size": 4000,
        "sample_batch_size": 200,
        "fuse_mode": SOFT_FUSE if args.soft else HARD_FUSE,
        "lr": 0.005,
        "evolution": {
            "train_batch_size": 4000,  # The same as PPO
            "num_workers": 10,  # default is 10,
            "optimizer_type": args.es_optimizer
        }
    }

    if args.ppo:
        config.pop("evolution")
        config.pop("fuse_mode")
        run = "PPO"
    else:
        run = EPTrainer

    if args.es:
        config = config["evolution"]
        config["env"] = env_name
        config["lr"] = 0.005
        config["episodes_per_batch"] = 1
        config["num_cpus_per_worker"] = 0.5
        run = "ES"

    ret = tune.run(
        run,
        local_dir=dir_path,
        name="DELETEME_NEW_IMPL_DICE",
        stop={"episode_reward_mean": 190},
        config=config,
        verbose=2,
        max_failures=0
    )
    shutil.rmtree(dir_path, ignore_errors=True)
    print("Test finished! Cost time: ", time.time() - now)

# Regression test result:
# Soft:
#     Reward: 191.21, Timesteps: 48000, Total time: 64.60, Iteration: 12,
#     Reported Time: 55.19
# Hard:
#     Reward: 192.67, Timesteps: 44000, Total time: 61.75, Iteration: 11,
#     Reported Time: 52.33
# PPO:
#     Reward: 192.93, Timesteps: 44000, Total time: 47.14, Iteration: 11,
#     Reported Time: 38.08
# ES:
#     Reward: 200, Timesteps: 1433880, Total time: 164.74, Iteration: 297,
#     Reported Time: 146.55
