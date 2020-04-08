import pytest
from ray import tune

from toolbox import initialize_ray
from toolbox.evolution_plugin.evolution_plugin import EPTrainer, filter_weights
import ray
true_false_pair = tune.grid_search([True, False])

DEFAULT_POLICY_NAME = "default_policy"


@pytest.fixture()
def ep_trainer():
    initialize_ray(test_mode=True, local_mode=True)
    return EPTrainer(env="BipedalWalker-v2", config=dict(
        num_sgd_iter=2,
        train_batch_size=400
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


def test_plugin_weight_sync(ep_trainer):
    assert_weights_equal(
        filter_weights(ep_trainer.get_policy().get_weights()),
        filter_weights(ray.get(
            ep_trainer._evolution_plugin.retrieve_weights.remote()))
    )

    ep_trainer.train()

    assert_weights_equal(
        filter_weights(ep_trainer.get_policy().get_weights()),
        filter_weights(ray.get(
            ep_trainer._evolution_plugin.retrieve_weights.remote()))
    )


if __name__ == "__main__":
    pytest.main(["-v"])
