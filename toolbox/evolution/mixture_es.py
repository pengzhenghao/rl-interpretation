from toolbox.action_distribution.mixture_gaussian import GaussianMixture
from toolbox.evolution.modified_es import GaussianESTrainer



if __name__ == '__main__':
    from toolbox import initialize_ray

    initialize_ray(test_mode=True, local_mode=True)
    config = {
        "num_workers": 3,
        "episodes_per_batch": 5,
        "train_batch_size": 150,
        "observation_filter": "NoFilter",
        "noise_size": 1000000,
        "model": {
            "custom_action_dist": GaussianMixture.name,
            "custom_options": {
                "num_components": 3
            }
        }
    }
    agent = GaussianESTrainer(config, "BipedalWalker-v2")

    agent.train()
