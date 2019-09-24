import scipy

from toolbox.visualize.draw_2d import draw_heatmap


def get_spearman(
        probe_mask, norm_next_obs_diff, batch_size=1000000, pval_th=0.05
):
    rho, pval = scipy.stats.spearmanr(
        probe_mask[:batch_size], norm_next_obs_diff[:batch_size], axis=0
    )

    mask = pval < pval_th  # small p_val means greater confidence.

    print_mask = mask[:probe_mask.shape[1], probe_mask.shape[1]:]

    print(
        "We found {} of pairs that have 95% confidence. The proportion is"
        "{}.".format(
            print_mask.sum(),
            print_mask.sum() / (print_mask.shape[0] * print_mask.shape[1])
        )
    )

    return rho, pval, mask


def draw_spearman(rho, pval, mask, size):
    rho = rho * mask
    # size = mask.shape[1]
    draw_heatmap(rho[:size, size:], matrix2=pval[:size, size:])
