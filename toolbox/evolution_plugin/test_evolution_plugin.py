import pytest
import ray
from ray import tune
from ray.tune.logger import pretty_print

from toolbox import initialize_ray
from toolbox.evolution_plugin.evolution_plugin import EPTrainer
from toolbox.evolution_plugin.fuse_gradient import HARD_FUSE, SOFT_FUSE

true_false_pair = tune.grid_search([True, False])

DEFAULT_POLICY_NAME = "default_policy"


@pytest.fixture(params=[HARD_FUSE, SOFT_FUSE])
def ep_trainer(request):
    initialize_ray(test_mode=True, local_mode=False)
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


def test_plugin_step(ep_trainer):
    # Test 1: Assert plugin.step() change the weights
    plugin = ep_trainer._evolution_plugin
    old_weights = ray.get(plugin.get_weights.remote())
    train_result, new_weight, reported_old_weight = \
        ray.get(plugin.step.remote(True))
    assert old_weights["default_policy"] == pytest.approx(
        reported_old_weight["default_policy"])
    assert old_weights["default_policy"] != pytest.approx(
        new_weight)
    print("One evolution step finish {} episodes.".format(
        train_result["episodes_this_iter"]
    ))
    assert ep_trainer.get_policy()._variables.get_flat() == pytest.approx(
        ep_trainer._previous_master_weights
    )

    # Test 2: Assert trainer.train() is bug-free
    result = ep_trainer.train()
    assert ep_trainer.get_policy()._variables.get_flat() == pytest.approx(
        ep_trainer._previous_master_weights
    )

    print("Train result: ", pretty_print(result))


if __name__ == "__main__":
    pytest.main(["-v"])
