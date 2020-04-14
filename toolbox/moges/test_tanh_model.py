import numpy as np
import pytest

from toolbox import initialize_ray
from toolbox.action_distribution import GaussianMixture
from toolbox.evolution.modified_es import GaussianESTrainer
from toolbox.moges.fcnet_tanh import register_tanh_model

tanh_model = register_tanh_model()


@pytest.fixture(params=["zero", "normal", "free"])
def gaussian_mixture_trainer(request):
    initialize_ray(test_mode=True, local_mode=False)
    trainer = GaussianESTrainer(env="BipedalWalker-v2", config={
        "model": {
            "custom_model": tanh_model,
            "custom_action_dist": GaussianMixture.name,
            "custom_options": {
                "num_components": 2,
                "std_mode": request.param
            }},
        "num_workers": 1,
        "train_batch_size": 300,
        "sample_batch_size": 100
    })
    return trainer


def test_gaussian_mixture(gaussian_mixture_trainer):
    trainer = gaussian_mixture_trainer
    trainer.get_policy().compute_actions(np.zeros([24]))
    trainer.train()

    if trainer.config["model"]["custom_options"]["std_mode"] == "zero":
        assert np.all(trainer.get_policy().compute_actions(
            np.random.random([24]))[0][8:16] == 1)


if __name__ == '__main__':
    pytest.main(["-vv"])
