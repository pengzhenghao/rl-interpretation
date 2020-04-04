import argparse

from toolbox.dice import utils as old_const
from toolbox.dice.test_dice import _test_dice as _test_dice_old
from toolbox.dies.appo_impl.test_dice_appo import _test_dice

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", action="store_true")
    args = parser.parse_args()

    if args.old:
        _test_dice_old({
            old_const.ONLY_TNB: True,
            old_const.USE_DIVERSITY_VALUE_NETWORK: False,
            old_const.NORMALIZE_ADVANTAGE: True,
            old_const.TWO_SIDE_CLIP_LOSS: False,
            "lr": 0.01
        },
            num_agents=5,
            local_mode=False,
            t=20000
        )
    else:
        _test_dice({
            "train_batch_size": 4000,
            "sample_batch_size": 200,
            "num_workers": 5,
            "num_agents": 5,
            "lr": 0.01,
        }, t=100000, env_name="CartPole-v0")
