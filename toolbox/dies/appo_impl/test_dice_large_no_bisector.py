import argparse
import shutil
import tempfile
import time

from ray import tune

from toolbox import initialize_ray
from toolbox.dice import utils as old_const, DiCETrainer
from toolbox.dies.appo_impl.dice_trainer import DiCETrainer_APPO
from toolbox.marl import MultiAgentEnvWrapper

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", action="store_true")
    args = parser.parse_args()

    local_mode = True
    env_name = "MountainCarContinuous-v0"
    dir_path = tempfile.mkdtemp()
    now = time.time()

    initialize_ray(test_mode=False, local_mode=local_mode, num_gpus=1)

    if args.old:
        env_config = {"env_name": env_name, "num_agents": 5}
        config = {
            "env": MultiAgentEnvWrapper,
            "env_config": env_config,
            "num_gpus": 1,
            "num_sgd_iter": 10,
            "train_batch_size": 4000,
            "sample_batch_size": 200,
            "num_workers": 5,
            "num_envs_per_worker": 10,
            old_const.ONLY_TNB: True,
            old_const.USE_DIVERSITY_VALUE_NETWORK: False,
            old_const.NORMALIZE_ADVANTAGE: True,
            old_const.TWO_SIDE_CLIP_LOSS: False,
            old_const.USE_BISECTOR: False
        }
        ret = tune.run(
            DiCETrainer,
            local_dir=dir_path,
            name="DELETEME_OLD_IMPL_DICE",
            stop={"timesteps_total": 1000000},
            config=config,
            verbose=2,
            max_failures=0
        )
    else:
        config = {
            "env": env_name,
            # "num_gpus": 1,
            "train_batch_size": 100,
            "sample_batch_size": 20,
            "num_workers": 1,
            "num_agents": 1,
            "num_envs_per_worker": 1,
            old_const.USE_BISECTOR: False
        }
        ret = tune.run(
            DiCETrainer_APPO,
            local_dir=dir_path,
            name="DELETEME_NEW_IMPL_DICE",
            stop={"timesteps_total": 5000000},
            config=config,
            verbose=2,
            max_failures=0
        )
    shutil.rmtree(dir_path, ignore_errors=True)
    print("Test finished! Cost time: ", time.time() - now)
