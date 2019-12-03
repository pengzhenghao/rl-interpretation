import argparse
import pickle

from ray import tune

# from toolbox import initialize_ray, get_local_dir
from toolbox.cooperative_exploration.ceppo import CEPPOTrainer, \
    OPTIONAL_MODES
from toolbox.cooperative_exploration.cetd3 import CETD3Trainer, SHARE_SAMPLE
from toolbox.cooperative_exploration.test_cooperative_exploration import \
    _validate_base


def validate_ceppo(stop, env="CartPole-v0"):
    return _validate_base(
        {
            "mode": tune.grid_search(OPTIONAL_MODES),
            "num_cpus_per_worker": 2,
            "num_gpus": 0.4
        }, False, env,
        CEPPOTrainer, stop
    )


def validate_cetd3(stop, env="MountainCarContinuous-v0"):
    return _validate_base(
        {
            "mode": tune.grid_search([SHARE_SAMPLE, None]),
            "num_gpus": 0.4,
            "num_cpus_for_driver": 2
        }, False,
        env, CETD3Trainer, stop
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", required=True, type=str)
    parser.add_argument("--number", "-n", default=1000000, type=str)
    args = parser.parse_args()

    if args.trainer == "td3":
        ret = validate_cetd3(args.number)
    elif args.trainer == "ppo":
        ret = validate_ceppo(args.number)
    else:
        raise ValueError()

    with open("1203-mountaincar_{}_{}steps_result.pkl".format(
            args.trainer, args.number), "wb") as f:
        d = ret.fetch_trial_dataframes()
        pickle.dump(d, f)
