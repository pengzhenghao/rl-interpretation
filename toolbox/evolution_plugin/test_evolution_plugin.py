import pickle

import numpy as np
import pytest
import ray
from ray import tune
from ray.tune.logger import pretty_print

from toolbox import initialize_ray
from toolbox.evolution_plugin.evolution_plugin import EPTrainer, _flatten, \
    _unflatten, _get_diff
from toolbox.evolution_plugin.fuse_gradient import HARD_FUSE, SOFT_FUSE

true_false_pair = tune.grid_search([True, False])

DEFAULT_POLICY_NAME = "default_policy"


@pytest.fixture(params=[HARD_FUSE, SOFT_FUSE])
def ep_trainer(request):
    initialize_ray(test_mode=True, local_mode=True)
    return EPTrainer(env="BipedalWalker-v2", config=dict(
        num_sgd_iter=2,
        train_batch_size=400,
        evolution=dict(
            episodes_per_batch=20,
            train_batch_size=400,
            noise_size=20000000
        ),
        fuse_mode=request.param
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


def test_getstate(ep_trainer):
    ep_trainer.train()
    path = ep_trainer.save()
    with open(path, "rb") as f:
        state = pickle.load(f)
    assert "plugin" in state["trainer_state"]
    assert isinstance(state["trainer_state"]["plugin"], dict)


def test_plugin_step(ep_trainer):
    # Test 1: Assert plugin.step() change the weights
    plugin = ep_trainer._evolution_plugin
    old_weights = ray.get(plugin.get_weights.remote())
    train_result, new_weight, reported_old_weight = \
        ray.get(plugin.step.remote(True))
    assert old_weights["default_policy"] == pytest.approx(
        _flatten(reported_old_weight)[0])
    print("One evolution step finish {} episodes.".format(
        train_result["episodes_this_iter"]
    ))
    assert_weights_equal(
        ep_trainer.get_policy().get_weights(),
        ep_trainer._previous_master_weights)

    # Test 2: Assert trainer.train() is bug-free
    result = ep_trainer.train()
    assert_weights_equal(
        ep_trainer.get_policy().get_weights(),
        ep_trainer._previous_master_weights)

    print("Train result: ", pretty_print(result))


def _check_shapes(shapes1, shapes2):
    # TODO remove this function
    for (n1, s1), (n2, s2) in zip(shapes1.items(), shapes2.items()):
        assert n1 == n2
        assert s1 == s2


@pytest.fixture()
def fake_weights():
    return {
        "fc1": np.random.randint(-10, 10, size=[100, 24]),
        "fc2": np.random.randint(-10, 10, size=[50, 77]),
        "fc1_value": np.random.randint(-10, 10, size=[50, 77]),
        "fc2_value": np.random.randint(-10, 10, size=[50, 77]),
    }


def test_weight_processing(fake_weights):
    flat, shapes = _flatten(fake_weights)
    assert isinstance(flat, np.ndarray)
    assert flat.ndim == 1

    recovered = _unflatten(flat, shapes)
    assert_weights_equal(recovered, fake_weights)

    diff, shapes = _get_diff(fake_weights, fake_weights)
    assert isinstance(diff, np.ndarray)
    assert diff.ndim == 1
    assert diff.size == (100 * 24 + 50 * 77)
    assert diff.size == sum(np.prod(s) for s in shapes.values())


if __name__ == "__main__":
    pytest.main(["-v"])
