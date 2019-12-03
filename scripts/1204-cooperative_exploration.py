import argparse
import pickle

from ray import tune

from toolbox.cooperative_exploration.ceppo import CEPPOTrainer, OPTIONAL_MODES
from toolbox.cooperative_exploration.cetd3 import CETD3Trainer, SHARE_SAMPLE
from toolbox.cooperative_exploration.test_cooperative_exploration import \
    _validate_base


def train_ceppo(test=False):
    return _validate_base(
        {
            "seed": tune.grid_search([100, 200, 300]),
            "num_sgd_iter": 10,
            "num_envs_per_worker": 16,
            "gamma": 0.99,
            "entropy_coeff": 0.001,
            "lambda": 0.95,
            "lr": 2.5e-4,
            "mode": tune.grid_search(OPTIONAL_MODES),
            "num_gpus": 0.3
        },
        test_mode=False,
        env_name="BipedalWalker-v2",
        trainer=CEPPOTrainer,
        stop=int(10e6) if not test else 1000,
        name="1204-ceppo-bipedalwalker" if not test else "DELETEME-1204-ceppo",
        num_gpus=8
    )


def train_cetd3(test=False):
    return _validate_base(
        {
            "seed": tune.grid_search([100, 200, 300]),
            "mode": tune.grid_search([SHARE_SAMPLE, None]),
            "num_gpus": 1,
            "num_cpus_for_driver": 2
        },
        test_mode=False,
        env_name="BipedalWalker-v2",
        trainer=CETD3Trainer,
        stop=int(10e6) if not test else 1000,
        name="1204-cetd3-bipedalwalker" if not test else "DELETEME-1204-cetd3",
        num_gpus=8
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", required=True, type=str)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.trainer == "td3":
        ret = train_cetd3(args.test)
    elif args.trainer == "ppo":
        ret = train_ceppo(args.test)
    else:
        raise ValueError()

    path = "data/1204-bipedalwalker-{}-{}steps-result.pkl".format(args.trainer,
                                                                  args.number)
    with open(path, "wb") as f:
        d = ret.fetch_trial_dataframes()
        pickle.dump(d, f)
        print("Result is saved at: <{}>".format(path))
