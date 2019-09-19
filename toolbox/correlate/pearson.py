import numpy as np
import scipy
import time


def get_pearson_correlation_coefficient(probe_mask, new_obs):
    sample_size = probe_mask.shape[0]
    assert sample_size == new_obs.shape[0]
    mask_length = probe_mask.shape[1]
    obs_length = new_obs.shape[1]

    rho_matrix = np.zeros((mask_length, obs_length))
    p_matrix = np.zeros((mask_length, obs_length))

    start = now = time.time()

    for mask_id in range(mask_length):
        print("[T +{:.2f}s/{:.2f}s]Current working row: {}".format(
            time.time()-now, time.time()-start, mask_id)
        )
        now = time.time()
        for obs_id in range(obs_length):
            rho, pval = scipy.stats.pearsonr(probe_mask[:, mask_id],
                                             new_obs[:, obs_id])
            rho_matrix[mask_id, obs_id] = rho
            p_matrix[mask_id, obs_id] = pval
    return rho_matrix, p_matrix
