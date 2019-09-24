import time

import numpy as np
import scipy

from toolbox.visualize.draw_2d import draw_heatmap


def get_pearson(probe_mask, new_obs, batch_size=1000000, pval_th=0.05):
    sample_size = probe_mask.shape[0]
    assert sample_size == new_obs.shape[0]
    assert sample_size >= batch_size
    mask_length = probe_mask.shape[1]
    obs_length = new_obs.shape[1]

    rho_matrix = np.zeros((mask_length, obs_length))
    p_matrix = np.zeros((mask_length, obs_length))

    start = now = time.time()

    for mask_id in range(mask_length):
        print(
            "[T +{:.2f}s/{:.2f}s]Current working row: {}".format(
                time.time() - now,
                time.time() - start, mask_id
            )
        )
        now = time.time()
        for obs_id in range(obs_length):
            rho, pval = scipy.stats.pearsonr(
                probe_mask[:batch_size, mask_id], new_obs[:batch_size, obs_id]
            )
            rho_matrix[mask_id, obs_id] = rho
            p_matrix[mask_id, obs_id] = pval

    print_mask = p_matrix < pval_th
    print(
        "We found {} of pairs that have 95% confidence. The proportion is"
        "{}.".format(
            print_mask.sum(),
            print_mask.sum() / (print_mask.shape[0] * print_mask.shape[1])
        )
    )

    return rho_matrix, p_matrix, print_mask


# get_pearson = get_pearson_correlation_coefficient


def draw_pearson(rho, pval, mask):
    rho = rho * mask
    draw_heatmap(rho, matrix2=pval)
