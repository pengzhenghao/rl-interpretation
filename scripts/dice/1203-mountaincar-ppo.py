import argparse

from ray import tune

from toolbox.cooperative_exploration.ceppo import CEPPOTrainer, OPTIONAL_MODES
from toolbox.cooperative_exploration.test_cooperative_exploration import \
    _validate_base


def train_atari(test=False):
    env_name = tune.grid_search([
        "BreakoutNoFrameskip-v4", "BeamRiderNoFrameskip-v4",
        "QbertNoFrameskip-v4", "SpaceInvadersNoFrameskip-v4"]
    )

    return _validate_base(
        {
            "lambda": 0.95,
            "kl_coeff": 0.5,
            "clip_rewards": True,
            "clip_param": 0.3,
            "entropy_coeff": 0.01,
            "train_batch_size": 5000,
            "sample_batch_size": 100,
            "sgd_minibatch_size": 500,
            "num_sgd_iter": 10,
            "num_workers": 10,
            "num_envs_per_worker": 5,
            "vf_share_layers": True,

            "mode": tune.grid_search(OPTIONAL_MODES),
            "num_gpus": 0.45
        },
        test_mode=test,
        env_name=env_name,
        trainer=CEPPOTrainer,
        stop=int(10e6) if not test else 1000,
        name="DELETEME_TEST" if test else "1203-mountaincar-ppo",
        num_gpus=8
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    train_atari(args.test)
