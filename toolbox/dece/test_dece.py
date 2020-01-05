from ray import tune

from toolbox.dece.dece import DECETrainer
from toolbox.dece.utils import *
from toolbox.marl.test_extra_loss import _base
from toolbox.env import FourWayGridWorld


def test_dece(config={}, local_mode=False):
    _base(
        trainer=DECETrainer,
        local_mode=local_mode,
        extra_config=config,
        env_name="Pendulum-v0",
        t=2000
    )


def test_dece_batch0(local_mode=False):
    test_dece(
        {
            DIVERSITY_ENCOURAGING: True,
            USE_BISECTOR: tune.grid_search([True, False]),
            USE_DIVERSITY_VALUE_NETWORK: tune.grid_search([True, False]),
            CLIP_DIVERSITY_GRADIENT: True,
            DELAY_UPDATE: tune.grid_search([True, False]),
            REPLAY_VALUES: tune.grid_search([True, False]),
            TWO_SIDE_CLIP_LOSS: tune.grid_search([True, False])
        }, local_mode
    )


def test_two_side_loss(local_mode=False):
    test_dece(
        {TWO_SIDE_CLIP_LOSS: tune.grid_search([True, False])}, local_mode
    )


def test_delay_update(local_mode=False):
    test_dece({DELAY_UPDATE: tune.grid_search([True, False])}, local_mode)


def test_three_tuning(local_mode=False):
    test_dece(
        {
            DELAY_UPDATE: tune.grid_search([True, False]),
            USE_DIVERSITY_VALUE_NETWORK: tune.grid_search([True, False]),
            REPLAY_VALUES: tune.grid_search([True, False])
        }, local_mode
    )


def test_vtrace(local_mode=False):
    _base(
        trainer=DECETrainer,
        local_mode=local_mode,
        extra_config={
            'use_vtrace': True,
            'sample_batch_size': 50,
            'train_batch_size': 200,
            'num_sgd_iter': 10,
            'lr': 5e-3,
        },
        env_name=FourWayGridWorld,
        t=100000
    )


def test_vtrace_single_agent(local_mode=False):
    _base(
        trainer=DECETrainer,
        local_mode=local_mode,
        extra_config={
            'use_vtrace': True,
            'sample_batch_size': 50,
            'train_batch_size': 200,
            'num_sgd_iter': 10
        },
        env_name="BipedalWalker-v2",
        t=2000,
        num_agents=1
    )


if __name__ == '__main__':
    # test_dece(local_mode=False)
    # test_dece_batch0(local_mode=False)
    # test_two_side_loss(local_mode=True)
    # test_delay_update(local_mode=False)
    # test_three_tuning(local_mode=False)
    test_vtrace(local_mode=False)
    # test_vtrace_single_agent(local_mode=False)
