import numpy as np
import pytest
from ray.rllib.agents.pg import PGTrainer

from toolbox import initialize_ray
from toolbox.action_distribution import GaussianMixture, DeterministicMixture
from toolbox.evolution import GaussianESTrainer


@pytest.fixture(params=[1, 2, 3])
def gaussian_mixture_trainer(request):
    initialize_ray(test_mode=True, local_mode=False)
    trainer = PGTrainer(
        env="BipedalWalker-v2",
        config={
            "model": {
                "custom_action_dist": GaussianMixture.name,
                "custom_options": {
                    "num_components": request.param
                }
            },
            "num_workers": 0,
            "train_batch_size": 300,
            "sample_batch_size": 100
        }
    )
    return trainer


def test_gaussian_mixture(gaussian_mixture_trainer):
    trainer = gaussian_mixture_trainer
    trainer.get_policy().compute_actions(np.zeros([1, 24]))
    trainer.train()


@pytest.fixture(params=[1, 2, 3])
def deterministic_mixture_trainer(request):
    initialize_ray(test_mode=True, local_mode=True)
    trainer = GaussianESTrainer(
        env="BipedalWalker-v2",
        config={
            "model": {
                "custom_action_dist": DeterministicMixture.name,
                "custom_options": {
                    "num_components": request.param
                }
            },
            "num_workers": 1,
            "episodes_per_batch": 1,
            "train_batch_size": 300,
            "sample_batch_size": 100
        }
    )
    return trainer


def test_deterministic_mixture(deterministic_mixture_trainer):
    trainer = deterministic_mixture_trainer
    trainer.get_policy().compute_actions(np.zeros([24]))
    trainer.train()


if __name__ == '__main__':
    pytest.main(["-vv"])
