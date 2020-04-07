import argparse
import shutil
import tempfile
import time

from ray import tune

from toolbox import initialize_ray
from toolbox.ppo_pipeline.ppo_pipeline_trainer import PPOPipelineTrainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--not-sync-sampling", "-ns", action="store_true")
    parser.add_argument("--local-mode", action="store_true")
    args = parser.parse_args()

    local_mode = args.local_mode
    env_name = "CartPole-v0"
    # env_name = "MountainCarContinuous-v0"
    dir_path = tempfile.mkdtemp()
    now = time.time()

    initialize_ray(test_mode=True, local_mode=local_mode, num_gpus=0)

    config = {
        "env": env_name,
        # "num_gpus": 1,
        "train_batch_size": 100,
        "sample_batch_size": 20,
        "num_workers": 1,
        "num_agents": 1,
        "num_envs_per_worker": 1,
        "sync_sampling": not args.not_sync_sampling
    }
    ret = tune.run(
        PPOPipelineTrainer,
        local_dir=dir_path,
        name="DELETEME_NEW_IMPL_DICE",
        stop={"timesteps_total": 5000000, "episode_reward_mean": 190},
        config=config,
        verbose=2,
        max_failures=0
    )
    shutil.rmtree(dir_path, ignore_errors=True)
    print("Test finished! Cost time: ", time.time() - now)
