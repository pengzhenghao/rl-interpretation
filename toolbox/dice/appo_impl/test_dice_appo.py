import copy
import shutil
import tempfile
import unittest

import pytest
from ray import tune

from toolbox import initialize_ray
from toolbox.dice import utils
from toolbox.dice.appo_impl.dice_appo import DiCETrainer_APPO

num_agents_pair = tune.grid_search([1, 3])

true_false_pair = tune.grid_search([True, False])

DEFAULT_POLICY_NAME = "default_policy"


@pytest.fixture(params=[1, 3])
def dice_trainer(request):
    initialize_ray(test_mode=True, local_mode=True)
    return DiCETrainer_APPO(env="BipedalWalker-v2", config=dict(
        num_agents=request.param,
        delay_update=False
    ))


def assert_weights_equal(w1, w2):
    assert isinstance(w1, dict)
    assert isinstance(w2, dict)
    for (wid1, arr1), (wid2, arr2) in zip(
            w1.items(), w2.items()
    ):
        assert arr1 == pytest.approx(arr2)
        if wid1.startswith(DEFAULT_POLICY_NAME) and \
                wid2.startswith(DEFAULT_POLICY_NAME):
            assert wid1 == wid2


@pytest.mark.unittest
def test_policy_pool_sync(dice_trainer):
    init_policy_pool = copy.deepcopy(dice_trainer._central_policy_weights)

    # Assert policy map in all workerset are synced
    for _, ws in dice_trainer.workers.items():
        for (pid1, w1), (pid3, po), (pid4, po4) in zip(
                # ws.local_worker()._local_policy_weights.items(),
                init_policy_pool.items(),
                ws.local_worker()._local_policy_pool.items(),
                ws.local_worker().get_policy().policy_pool.items()
        ):
            # central weights equal to local weights
            # assert pid1 == pid2
            # assert_weights_equal(w1, w2)

            # central weights equal to local worker-owned pool's weights
            assert pid3 == pid1
            assert_weights_equal(w1, po.get_weights())

            # central weights equal to local policy-owned pool's weights
            assert pid4 == pid1
            assert_weights_equal(po.get_weights(), po4.get_weights())

    # Step forward
    dice_trainer.train()
    new_policy_pool = copy.deepcopy(dice_trainer._central_policy_weights)

    # Assert the policies is changed, so old one should not equal to the new
    for (pid1, w1), (pid2, w2) in zip(
            init_policy_pool.items(),
            new_policy_pool.items()
    ):
        assert pid1 == pid2
        for (wid1, arr1), (wid2, arr2) in zip(
                w1.items(), w2.items()
        ):
            assert arr1 != pytest.approx(arr2)
            assert wid1 == wid2

    # Assert policy map in all workerset are synced
    for _, ws in dice_trainer.workers.items():
        for (pid1, w1), (pid3, po), (pid4, po4) in zip(
                # ws.local_worker()._local_policy_weights.items(),
                new_policy_pool.items(),
                ws.local_worker()._local_policy_pool.items(),
                ws.local_worker().get_policy().policy_pool.items()
        ):
            # assert pid1 == pid2
            # assert_weights_equal(w1, w2)

            assert pid3 == pid1
            assert_weights_equal(w1, po.get_weights())

            # since the policy-owned policy pool only take the reference of
            # the local worker-owned policy pool, so they are automatically
            # synced.
            assert pid4 == pid1
            assert_weights_equal(po.get_weights(), po4.get_weights())


def _test_dice(
        extra_config={},
        local_mode=False,
        num_agents=3,
        env_name="BipedalWalker-v2",
        t=2000
):
    num_gpus = 0
    initialize_ray(test_mode=True, local_mode=local_mode, num_gpus=num_gpus)

    # default config
    # env_config = {"env_name": env_name, "num_agents": num_agents}
    config = {
        "env": "BipedalWalker-v2",

        "num_agents": num_agents,

        # "env": MultiAgentEnvWrapper,
        # "env_config": env_config,
        "num_gpus": num_gpus,
        "log_level": "DEBUG",
        "sample_batch_size": 20,
        "train_batch_size": 100,
        # "sgd_minibatch_size": 60,
        "num_sgd_iter": 3,
    }

    if extra_config:
        config.update(extra_config)
    stop = {"timesteps_total": t} if not isinstance(t, dict) else t
    dir_path = tempfile.mkdtemp()
    ret = tune.run(
        DiCETrainer_APPO,
        local_dir=dir_path,
        name="DELETEME_TEST_extra_loss_ppo_trainer",
        stop=stop,
        config=config,
        verbose=2,
        max_failures=0
    )
    shutil.rmtree(dir_path, ignore_errors=True)
    return ret


class DiCETest(unittest.TestCase):
    def test_use_bisector(self):
        _test_dice({utils.USE_BISECTOR: False}, num_agents=num_agents_pair)

    def test_use_dnv(self):
        _test_dice({utils.DELAY_UPDATE: True}, num_agents=num_agents_pair)

    def test_delay_update(self):
        _test_dice({utils.DELAY_UPDATE: False}, num_agents=num_agents_pair)

    def test_tsc_loss(self):
        _test_dice({utils.TWO_SIDE_CLIP_LOSS: False},
                   num_agents=num_agents_pair)

    def test_only_tnb(self):
        _test_dice({utils.ONLY_TNB: True}, num_agents=num_agents_pair)

    def test_normalize_adv(self):
        _test_dice({utils.NORMALIZE_ADVANTAGE: True},
                   num_agents=num_agents_pair)

    def test_default(self):
        _test_dice(num_agents=tune.grid_search([1, 3, 5]))


if __name__ == "__main__":
    # pytest.main(["-v", "-m", "unittest"])
    # unittest.main(verbosity=2)

    # print("===== 3 agents =====")
    # _test_dice(
    #     # num_agents=tune.grid_search([1, 3, 5]),
    #     num_agents=3,
    #     local_mode=False
    # )

    # print("===== 1 agents =====")
    # _test_dice(
    #     # num_agents=tune.grid_search([1, 3, 5]),
    #     num_agents=1,
    #     local_mode=True
    # )

    # pytest.main(["-v -m unittest test_dice_appo.py"])

    _test_dice(
        num_agents=10,
        local_mode=False,
        t=300000
    )
