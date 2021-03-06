import argparse
import shutil
import tempfile
import time

from ray import tune

from toolbox import initialize_ray
from toolbox.dice import utils as old_const
from toolbox.dies.appo_impl.dice_trainer import DiCETrainer_APPO

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, help="In ppo, appo, dice")
    parser.add_argument("--local-mode", action="store_true")
    parser.add_argument("--num-agents", type=int, default=1)
    args = parser.parse_args()

    print(args)

    local_mode = args.local_mode
    env_name = "CartPole-v0"
    dir_path = tempfile.mkdtemp()
    now = time.time()
    num_gpus = 0

    initialize_ray(test_mode=True, local_mode=local_mode, num_gpus=1)

    if args.algo in ["ppo", "PPO"]:
        config = {
            "env": env_name,
            "num_sgd_iter": 10,
            "train_batch_size": 4000,
            "sample_batch_size": 200,
            "num_workers": 1,
            "num_gpus": num_gpus,
            "num_envs_per_worker": 10,
            "sgd_minibatch_size": 200
        }
        ret = tune.run(
            "PPO",
            local_dir=dir_path,
            name="DELETEME_OLD_IMPL_DICE",
            stop={"episode_reward_mean": 190},
            config=config,
            verbose=2,
            max_failures=0
        )
    elif args.algo in ["appo", "APPO"]:
        config = {
            "env": env_name,
            "num_sgd_iter": 10,
            "train_batch_size": 4000,
            "sample_batch_size": 200,
            "num_workers": 1,
            "num_envs_per_worker": 10,
            "num_gpus": num_gpus,
            "vtrace": False,

            "model": {"vf_share_layers": False},
        }
        ret = tune.run(
            "APPO",
            local_dir=dir_path,
            name="DELETEME_OLD_IMPL_DICE",
            stop={"episode_reward_mean": 190},
            config=config,
            verbose=2,
            max_failures=0
        )
    elif args.algo in ["DICE", "dice", "DiCE"]:
        config = {
            "env": env_name,
            "num_sgd_iter": 10,
            "num_gpus": num_gpus,
            "train_batch_size": 4000,
            "sample_batch_size": 200,
            "num_workers": args.num_agents,
            "num_agents": args.num_agents,
            "num_envs_per_worker": 10,
            old_const.USE_BISECTOR: False,
            "lr": 5e-5,

            # PPO config
            "grad_clip": None,
            "vf_loss_coeff": 1.0,
            "entropy_coeff": 0.0,

            # Special setting for sync sampling mode
            "sync_sampling": True,
        }
        ret = tune.run(
            DiCETrainer_APPO,
            local_dir=dir_path,
            name="DELETEME_NEW_IMPL_DICE",
            stop={"episode_reward_mean": 190},
            config=config,
            verbose=2,
            max_failures=0
        )
    elif args.algo in ["ADICE", "adice", "ADiCE"]:
        config = {
            "env": env_name,
            "num_sgd_iter": 10,
            "num_gpus": num_gpus,
            "train_batch_size": 4000,
            "sample_batch_size": 200,
            "num_workers":  args.num_agents,
            "num_agents":  args.num_agents,
            "num_envs_per_worker": 10,
            old_const.USE_BISECTOR: False,

            # Special setting for sync sampling mode
            "sync_sampling": False,
        }
        ret = tune.run(
            DiCETrainer_APPO,
            local_dir=dir_path,
            name="DELETEME_NEW_IMPL_DICE",
            stop={"episode_reward_mean": 190},
            config=config,
            verbose=2,
            max_failures=0
        )
    else:
        raise ValueError()
    shutil.rmtree(dir_path, ignore_errors=True)
    print("Test finished! Cost time: ", time.time() - now)
