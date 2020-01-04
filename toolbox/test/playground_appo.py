"""APPO has some interesting feature. We try to undertstand it by playing with
it in this file."""
from ray import tune

from toolbox import initialize_ray

if __name__ == '__main__':
    initialize_ray(local_mode=True, test_mode=True, num_gpus=0)
    tune.run(
        'APPO',
        config={'env': 'BipedalWalker-v2', 'vtrace': True, 'num_gpus': 0},
        stop={'timesteps_total': 10000}
    )
