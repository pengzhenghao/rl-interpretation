from ray import tune

from toolbox.ipd.tnb_rllib import TNBTrainer
from toolbox import initialize_ray


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


if __name__ == '__main__':
    test_train_ipd(local_mode=True)
