"""
This file implement the logic of switching
"""
import numpy as np

# Available fuse mode  (ED: evolution diff, MD: master diff)
# Note that we should conduct grad normalization before doing anything.
# Hard mode: if it's obtuse angle between ED and MD, use ED, else use MD.
HARD_FUSE = "hard_fuse"
# Soft mode: use TNB-style fusing
SOFT_FUSE = "soft_fuse"


def _clip_grad(grad, max_grad_norm):
    grad_norm = np.linalg.norm(grad)
    ret_grad = grad / grad_norm * min(grad_norm, max_grad_norm)
    return ret_grad, grad_norm


def fuse_gradient(
        master_diff,
        evolution_diff,
        fuse_mode,
        max_grad_norm=None,
        equal_norm=False
):
    assert isinstance(master_diff, np.ndarray)
    assert isinstance(evolution_diff, np.ndarray)
    assert master_diff.ndim == 1
    assert evolution_diff.ndim == 1

    # Gradient Normalization before doing anything
    if max_grad_norm is not None:
        master_diff, md_norm = _clip_grad(master_diff, max_grad_norm)
        evolution_diff, ed_norm = _clip_grad(evolution_diff, max_grad_norm)
    else:
        md_norm = np.linalg.norm(master_diff)
        ed_norm = np.linalg.norm(evolution_diff)

    # Get the cos of two vectors
    master_diff_norm = master_diff / md_norm
    evolution_diff_norm = evolution_diff / ed_norm
    cos = np.dot(master_diff_norm, evolution_diff_norm)
    assert -1 <= cos <= 1

    # Make two differences to have same norm, if required
    if equal_norm:
        middle_norm = (md_norm + ed_norm) / 2
        master_diff = master_diff_norm * middle_norm
        evolution_diff = evolution_diff_norm * middle_norm

    if fuse_mode == HARD_FUSE:
        if cos > 0:  # acute angle
            return_diff = master_diff
        else:  # obtuse angle
            return_diff = evolution_diff

    elif fuse_mode == SOFT_FUSE:
        # Compute the bisector unit vector
        bisector = master_diff_norm + evolution_diff_norm
        bisector = bisector / np.linalg.norm(bisector)

        # Compute the mean projection length
        master_proj_length = np.linalg.norm(np.dot(master_diff, bisector))
        evolution_proj_length = np.linalg.norm(
            np.dot(evolution_diff, bisector)
        )
        bisector = bisector * (master_proj_length + evolution_proj_length) / 2
        return_diff = bisector

    else:
        raise ValueError(
            "Your input fuse_mode {} not in {}.".format(
                fuse_mode, [HARD_FUSE, SOFT_FUSE]
            )
        )

    stats = dict(
        master_diff_norm=md_norm,
        evolution_diff_norm=ed_norm,
        cosine=cos,
        fuse_mode=fuse_mode,
        fuse_diff_norm=np.linalg.norm(return_diff)
    )
    return return_diff, stats
