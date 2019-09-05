"""
This codes is mostly copied from google implementation of CKA.

https://github.com/google-research/google-research/tree/master
/representation_similarity

"""
import pickle

import numpy as np

from process_data import read_yaml
from rollout import several_agent_rollout
from utils import initialize_ray, get_random_string

ACTIVATION_DATA_PREFIX = "layer"


def gram_linear(x):
    """Compute Gram (kernel) matrix for a linear kernel.

    Args:
      x: A num_examples x num_features matrix of features.

    Returns:
      A num_examples x num_examples Gram matrix of examples.
    """
    return x.dot(x.T)


def gram_rbf(x, threshold=1.0):
    """Compute Gram (kernel) matrix for an RBF kernel.

    Args:
      x: A num_examples x num_features matrix of features.
      threshold: Fraction of median Euclidean distance to use as RBF kernel
        bandwidth. (This is the heuristic we use in the paper. There are other
        possible ways to set the bandwidth; we didn't try them.)

    Returns:
      A num_examples x num_examples Gram matrix of examples.
    """
    dot_products = x.dot(x.T)  # XX'
    sq_norms = np.diag(dot_products)  # tr(XX')

    #
    sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
    sq_median_distance = np.median(sq_distances)
    return np.exp(-sq_distances / (2 * threshold**2 * sq_median_distance))


def center_gram(gram, unbiased=False):
    """Center a symmetric Gram matrix.

    This is equvialent to centering the (possibly infinite-dimensional)
    features
    induced by the kernel before computing the Gram matrix.

    Args:
      gram: A num_examples x num_examples symmetric matrix.
      unbiased: Whether to adjust the Gram matrix in order to compute an
      unbiased
        estimate of HSIC. Note that this estimator may be negative.

    Returns:
      A symmetric matrix with centered columns and rows.
    """
    if not np.allclose(gram, gram.T):
        raise ValueError('Input must be a symmetric matrix.')
    gram = gram.copy()

    if unbiased:
        # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
        # L. (2014). Partial distance correlation with methods for
        # dissimilarities.
        # The Annals of Statistics, 42(6), 2382-2412, seems to be more
        # numerically
        # stable than the alternative from Song et al. (2007).
        n = gram.shape[0]
        np.fill_diagonal(gram, 0)
        means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
        means -= np.sum(means) / (2 * (n - 1))
        gram -= means[:, None]
        gram -= means[None, :]
        np.fill_diagonal(gram, 0)
    else:
        means = np.mean(gram, 0, dtype=np.float64)
        means -= np.mean(means) / 2
        gram -= means[:, None]
        gram -= means[None, :]

    return gram


def cka(gram_x, gram_y, debiased=False):
    """Compute CKA.

    Args:
      gram_x: A num_examples x num_examples Gram matrix.
      gram_y: A num_examples x num_examples Gram matrix.
      debiased: Use unbiased estimator of HSIC. CKA may still be biased.

    Returns:
      The value of CKA between X and Y.
    """
    gram_x = center_gram(gram_x, unbiased=debiased)
    gram_y = center_gram(gram_y, unbiased=debiased)

    # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased
    # variant) or
    # n*(n-3) (unbiased variant), but this cancels for CKA.
    scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

    normalization_x = np.linalg.norm(gram_x)
    normalization_y = np.linalg.norm(gram_y)
    return scaled_hsic / (normalization_x * normalization_y)


def _debiased_dot_product_similarity_helper(
        xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x,
        squared_norm_y, n
):
    """Helper for computing debiased dot product similarity (i.e. linear
    HSIC)."""
    # This formula can be derived by manipulating the unbiased estimator from
    # Song et al. (2007).
    return (
        xty - n / (n - 2.) * sum_squared_rows_x.dot(sum_squared_rows_y) +
        squared_norm_x * squared_norm_y / ((n - 1) * (n - 2))
    )


def feature_space_linear_cka(features_x, features_y, debiased=False):
    """Compute CKA with a linear kernel, in feature space.

    This is typically faster than computing the Gram matrix when there are
    fewer
    features than examples.

    Args:
      features_x: A num_examples x num_features matrix of features.
      features_y: A num_examples x num_features matrix of features.
      debiased: Use unbiased estimator of dot product similarity. CKA may
      still be biased. Note that this estimator may be negative.

    Returns:
      The value of CKA between X and Y.
    """
    features_x = features_x - np.mean(features_x, 0, keepdims=True)
    features_y = features_y - np.mean(features_y, 0, keepdims=True)

    dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y))**2
    normalization_x = np.linalg.norm(features_x.T.dot(features_x))
    normalization_y = np.linalg.norm(features_y.T.dot(features_y))

    if debiased:
        n = features_x.shape[0]
        # Equivalent to np.sum(features_x ** 2, 1) but avoids an
        # intermediate array.
        sum_squared_rows_x = np.einsum('ij,ij->i', features_x, features_x)
        sum_squared_rows_y = np.einsum('ij,ij->i', features_y, features_y)
        squared_norm_x = np.sum(sum_squared_rows_x)
        squared_norm_y = np.sum(sum_squared_rows_y)

        dot_product_similarity = _debiased_dot_product_similarity_helper(
            dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
            squared_norm_x, squared_norm_y, n
        )
        normalization_x = np.sqrt(
            _debiased_dot_product_similarity_helper(
                normalization_x**2, sum_squared_rows_x, sum_squared_rows_x,
                squared_norm_x, squared_norm_x, n
            )
        )
        normalization_y = np.sqrt(
            _debiased_dot_product_similarity_helper(
                normalization_y**2, sum_squared_rows_y, sum_squared_rows_y,
                squared_norm_y, squared_norm_y, n
            )
        )

    return dot_product_similarity / (normalization_x * normalization_y)


get_cka_result = feature_space_linear_cka


def cca(features_x, features_y):
    """Compute the mean squared CCA correlation (R^2_{CCA}).

    Args:
      features_x: A num_examples x num_features matrix of features.
      features_y: A num_examples x num_features matrix of features.

    Returns:
      The mean squared CCA correlations between X and Y.
    """
    qx, _ = np.linalg.qr(features_x)  # Or use SVD with full_matrices=False.
    qy, _ = np.linalg.qr(features_y)
    return np.linalg.norm(qx.T.dot(qy)
                          )**2 / min(features_x.shape[1], features_y.shape[1])


def test_origin():
    np.random.seed(1337)
    X = np.random.randn(100, 10)
    Y = np.random.randn(100, 10) + X

    cka_from_examples = cka(gram_linear(X), gram_linear(Y))
    cka_from_features = feature_space_linear_cka(X, Y)

    print('Linear CKA from Examples: {:.5f}'.format(cka_from_examples))
    print('Linear CKA from Features: {:.5f}'.format(cka_from_features))
    np.testing.assert_almost_equal(cka_from_examples, cka_from_features)

    rbf_cka = cka(gram_rbf(X, 0.5), gram_rbf(Y, 0.5))
    print('RBF CKA: {:.5f}'.format(rbf_cka))

    cka_from_examples_debiased = cka(
        gram_linear(X), gram_linear(Y), debiased=True
    )
    cka_from_features_debiased = feature_space_linear_cka(X, Y, debiased=True)

    print(
        'Linear CKA from Examples (Debiased): {:.5f}'.
        format(cka_from_examples_debiased)
    )
    print(
        'Linear CKA from Features (Debiased): {:.5f}'.
        format(cka_from_features_debiased)
    )

    np.testing.assert_almost_equal(
        cka_from_examples_debiased, cka_from_features_debiased
    )

    print('Mean Squared CCA Correlation: {:.5f}'.format(cca(X, Y)))

    transform = np.random.randn(10, 10)
    _, orthogonal_transform = np.linalg.eigh(transform.T.dot(transform))

    # CKA is invariant only to orthogonal transformations.
    np.testing.assert_almost_equal(
        feature_space_linear_cka(X, Y),
        feature_space_linear_cka(X.dot(orthogonal_transform), Y)
    )
    np.testing.assert_(
        not np.isclose(
            feature_space_linear_cka(X, Y),
            feature_space_linear_cka(X.dot(transform), Y)
        )
    )

    # CCA is invariant to any invertible linear transform.
    np.testing.assert_almost_equal(
        cca(X, Y), cca(X.dot(orthogonal_transform), Y)
    )
    np.testing.assert_almost_equal(cca(X, Y), cca(X.dot(transform), Y))

    # Both CCA and CKA are invariant to isotropic scaling.
    np.testing.assert_almost_equal(cca(X, Y), cca(X * 1.337, Y))
    np.testing.assert_almost_equal(
        feature_space_linear_cka(X, Y), feature_space_linear_cka(X * 1.337, Y)
    )


def build_agent_dataset(
        yaml_path, num_rollouts=100, seed=0, num_workers=10, output_path=None
):
    """
    return shape: {
        agent1: {
            obs: array,
            act: array,
            layer0: array,
            ...
        }
    }
    """
    data_dict = several_agent_rollout(
        yaml_path, num_rollouts, seed, num_workers, return_data=True
    )
    name_ckpt_mapping = read_yaml(yaml_path)
    agent_dataset = {}
    traj_count = 0
    for name, traj_list in data_dict.items():
        # traj_list[0].keys() = dict_keys(['t', 'eps_id', 'agent_index',
        # 'obs', 'actions', 'rewards', 'prev_actions', 'prev_rewards',
        # 'dones', 'infos', 'new_obs', 'action_prob', 'vf_preds',
        # 'behaviour_logits', 'layer0', 'layer1', 'unroll_id', 'advantages',
        # 'value_targets'])
        layer_keys = [
            k for k in traj_list[0].keys()
            if k.startswith(ACTIVATION_DATA_PREFIX)
        ]
        obs_list = [traj['obs'] for traj in traj_list]
        act_list = [traj['actions'] for traj in traj_list]
        rew_list = [traj['rewards'] for traj in traj_list]
        logits_list = [traj['behaviour_logits'] for traj in traj_list]

        agent_dict = {
            # each entry to list is A ROLLOUT
            "obs": np.concatenate(obs_list),
            "act": np.concatenate(act_list),
            "rew_list": np.concatenate(rew_list),
            "logit": np.concatenate(logits_list),
            "agent_info": name_ckpt_mapping[name]
        }
        for layer_key in layer_keys:
            layer_activation = [traj[layer_key] for traj in traj_list]
            agent_dict[layer_key] = np.concatenate(layer_activation)

        traj_count += sum([len(obs) for obs in obs_list])
        agent_dataset[name] = agent_dict

    print(
        "Successfully collect {} trajectories from {} agents!".format(
            traj_count, len(data_dict)
        )
    )

    if output_path is not None:
        if output_path.endswith(".pkl"):
            output_path = output_path.split(".pkl")[0]
        output_path = "{}_rollout{}_seed{}.pkl".format(
            output_path, num_rollouts, seed
        )
        with open(output_path, "wb") as f:
            pickle.dump(agent_dataset, f)
        print("agent_dataset is successfully saved at {}.".format(output_path))
    return agent_dataset


def sample_from_agent_dataset(agent_dataset, seed, batch_size=100):
    # agent_data_mappin = {agent: [[traj1:[t0], [t1], ..], [traj2], ...}
    # build a POOL of Observation
    # Our guide line is, concatenate all traj of a agent and
    # sample 100 (or maybe 1000) observations from each agent's obs pool.
    # so that we have 30,000 obs given 300 agents.
    """
    return shape: {
        agent1: {
            obs: array [batch_size, ..],
            act: array [batch_size, ..],
            layer0: array [batch_size, ..],
            ...
        }
    }
    """
    # np.random.seed(seed)
    rs = np.random.RandomState(seed)
    sample_agent_dataset = {}

    for agent_name, agent_dict in agent_dataset.items():
        agent_new_dict = {}
        agent_total_steps = agent_dict['obs'].shape[0]
        indices = rs.randint(0, agent_total_steps, batch_size)
        agent_new_dict['index'] = indices
        for data_name, data_array in agent_dict.items():
            if not isinstance(data_array, np.ndarray):
                agent_new_dict[data_name] = data_array
            else:
                batch = data_array[indices]
                agent_new_dict[data_name] = batch

        sample_agent_dataset[agent_name] = agent_new_dict

    return sample_agent_dataset


def build_obs_pool(sample_agent_dataset):
    obs_pool = []
    for agent_name, agent_dict in sample_agent_dataset.items():
        obs_pool.append(agent_dict['obs'])
    return np.concatenate(obs_pool)


def get_result():
    pass


from rollout import several_agent_replay


def test_new_implementation():
    yaml_path = "yaml/test-2-agents.yaml"
    agent_dataset = build_agent_dataset(
        yaml_path, 2, output_path="/tmp/{}".format(get_random_string())
    )
    sample_agent_dataset = sample_from_agent_dataset(agent_dataset, 0)
    obs_pool = build_obs_pool(sample_agent_dataset)
    ret = several_agent_replay(yaml_path, obs_pool)
    return ret


if __name__ == '__main__':
    # test_origin()
    initialize_ray(test_mode=True)
    agent_data = test_new_implementation()
