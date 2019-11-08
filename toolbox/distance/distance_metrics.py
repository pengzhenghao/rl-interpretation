import numpy as np


def _get_kl_divergence(dist1, dist2):
    assert dist1.ndim == 2
    assert dist2.ndim == 2

    source_mean, source_log_std = np.split(dist1, 2, axis=1)
    target_mean, target_log_std = np.split(dist2, 2, axis=1)

    kl_divergence = np.sum(
        target_log_std - source_log_std +
        (np.square(source_log_std) + np.square(source_mean - target_mean)) /
        (2.0 * np.square(target_log_std) + 1e-9) - 0.5,
        axis=1
    )
    kl_divergence = np.clip(kl_divergence, 0.0, 1e38)  # to avoid inf
    averaged_kl_divergence = np.mean(kl_divergence)
    return averaged_kl_divergence


def _build_matrix(iterable, apply_function, default_value=0):
    """
    Copied from toolbox.interface.cross_agent_analysis
    """
    length = len(iterable)
    matrix = np.empty((length, length))
    matrix.fill(default_value)
    for i1 in range(length - 1):
        for i2 in range(i1, length):
            repr1 = iterable[i1]
            repr2 = iterable[i2]
            result = apply_function(repr1, repr2)
            matrix[i1, i2] = result
            matrix[i2, i1] = result
    return matrix


def js_distance(action_list):
    num_agents = len(action_list)

    num_samples = action_list[0].shape[0] / num_agents
    # num_samples should be integer
    assert action_list[0].shape[0] % num_agents == 0, (
        action_list, [i.shape for i in action_list], action_list[0].shape,
        num_agents)
    num_samples = int(num_samples)

    js_matrix = np.zeros((num_agents, num_agents))

    for i1 in range(len(action_list) - 1):
        source = action_list[i1][i1 * num_samples:(i1 + 1) * num_samples]
        for i2 in range(i1, len(action_list)):
            target = action_list[i2][i2 * num_samples:(i2 + 1) * num_samples]
            average_distribution_source = \
                (source +
                 action_list[i2][i1 * num_samples: (i1 + 1) * num_samples]
                 ) / 2
            average_distribution_target = \
                (target +
                 action_list[i1][i2 * num_samples: (i2 + 1) * num_samples]
                 ) / 2

            js_divergence = _get_kl_divergence(
                source, average_distribution_source
            ) + _get_kl_divergence(target, average_distribution_target)

            js_divergence = js_divergence / 2
            js_matrix[i1, i2] = js_divergence
            js_matrix[i2, i1] = js_divergence
    js_matrix = np.sqrt(js_matrix)
    return js_matrix


def joint_dataset_distance(action_list):
    apply_function = lambda x, y: np.linalg.norm(x - y)
    dist_matrix = _build_matrix(action_list, apply_function)
    return dist_matrix
