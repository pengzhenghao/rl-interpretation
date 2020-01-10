from ray import tune

from toolbox.dece.dece import DECETrainer
from toolbox.dece.utils import *
from toolbox.env import FourWayGridWorld
from toolbox.marl.test_extra_loss import _base


def test_dece(config={}, local_mode=False, **kwargs):
    _base(
        trainer=DECETrainer,
        local_mode=local_mode,
        extra_config=config,
        env_name="Pendulum-v0",
        t=2000,
        **kwargs
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


def test_vtrace(local_mode=False, hard=False):
    _base(
        trainer=DECETrainer,
        local_mode=local_mode,
        extra_config={
            REPLAY_VALUES: True,
            'sample_batch_size': 50 if hard else 8,
            'train_batch_size': 200 if hard else 32,
            'num_sgd_iter': 10 if hard else 2,
            'model': {
                'fcnet_hiddens': [16, 16]
            },
            'seed': tune.grid_search([0, 100]) if hard else 0
            # 'lr': 5e-3,
        },
        env_name=FourWayGridWorld,
        t=100000
    )


def test_vtrace_single_agent(local_mode=False):
    _base(
        trainer=DECETrainer,
        local_mode=local_mode,
        extra_config={
            REPLAY_VALUES: True,
            'sample_batch_size': 50,
            'train_batch_size': 200,
            'num_sgd_iter': 10
        },
        env_name="BipedalWalker-v2",
        t=2000,
        num_agents=1
    )


def regression_test(local_mode=False):
    _base(
        trainer=DECETrainer,
        local_mode=local_mode,
        extra_config={
            REPLAY_VALUES: tune.grid_search([True, False]),
            # "normalize_advantage": tune.grid_search([True, False]),
            # 'use_vtrace': tune.grid_search([True]),
            'sample_batch_size': 128,
            'train_batch_size': 512,
            'sgd_minibatch_size': 32,
            'num_sgd_iter': 10,
            USE_BISECTOR: False,
            'seed': tune.grid_search([432, 1920]),
            # 'lr': 5e-3,
        },
        env_name="Pendulum-v0",
        # env_name="CartPole-v0",
        # env_name=FourWayGridWorld,
        t={'timesteps_total': 300000},
        num_agents=1
    )


def only_tnb(local_mode=False):
    test_dece(
        {
            DELAY_UPDATE: tune.grid_search([True, False]),
            ONLY_TNB: True,
            REPLAY_VALUES: False
        }, local_mode
    )


def single_agent_cetnb(lm=False):
    test_dece(
        {
            DELAY_UPDATE: tune.grid_search([True, False]),
            REPLAY_VALUES: tune.grid_search([False]),
            USE_DIVERSITY_VALUE_NETWORK: tune.grid_search([True, False]),
        },
        lm,
        num_agents=tune.grid_search([1, 3])
    )


def replay_values_or_not_test(lm=False):
    test_dece(
        {
            # DELAY_UPDATE: tune.grid_search([True, False]),
            REPLAY_VALUES: tune.grid_search([True, False]),
            # USE_DIVERSITY_VALUE_NETWORK: tune.grid_search([True, False]),
        },
        lm,
        num_agents=tune.grid_search([1, 3])
    )


if __name__ == '__main__':
    # test_dece(local_mode=False)
    # test_dece_batch0(local_mode=False)
    # test_two_side_loss(local_mode=True)
    # test_delay_update(local_mode=False)
    # test_three_tuning(local_mode=False)
    # single_agent_cetnb()
    # only_tnb()
    regression_test(local_mode=False)
    # test_vtrace(local_mode=True)
    # test_vtrace_single_agent(local_mode=True)
    # replay_values_or_not_test(False)
