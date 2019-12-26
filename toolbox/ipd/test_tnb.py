import ray
from ray import tune

from toolbox import initialize_ray
from toolbox.ipd.tnb import TNBTrainer
from toolbox.ipd.train_tnb import main as tnb_es


def test_train_ipd(local_mode=False):
    initialize_ray(test_mode=True, local_mode=local_mode)
    env_name = "CartPole-v0"
    # env_name = "BipedalWalker-v2"
    config = {"num_sgd_iter": 2, "env": env_name, "novelty_threshold": 0.5}
    tune.run(
        TNBTrainer,
        name="DELETEME_TEST",
        verbose=2,
        checkpoint_freq=10,
        checkpoint_at_end=True,
        stop={"timesteps_total": 50000},
        config=config
    )


def test_tnbes(local_mode=False):
    env_name = "CartPole-v0"
    config = {
        "num_sgd_iter": 2,
        "env": env_name,
        "use_preoccupied_agent": True
    }

    def ray_init():
        ray.shutdown()
        initialize_ray(test_mode=True, local_mode=local_mode, num_gpus=0)

    tnb_es(
        exp_name="DELETEME-TEST",
        num_iterations=10,
        max_num_agents=3,
        timesteps_total=30000,
        common_config=config,
        max_not_improve_iterations=3,
        ray_init=ray_init,
        test_mode=True
    )


if __name__ == '__main__':
    # test_train_ipd(local_mode=True)
    test_tnbes(local_mode=False)
