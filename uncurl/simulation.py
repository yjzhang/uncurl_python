# simulations... generating poisson data?

import numpy as np

def generate_poisson_data(centers, n_cells, cluster_probs=None):
    """
    Generates poisson-distributed data, given a set of means for each cluster.

    Args:
        centers (array): genes x clusters matrix
        n_cells (int): number of output cells
        cluster_probs (array): prior probability for each cluster.
            Default: uniform.

    Returns:
        array with shape genes x n_cells
    """
    genes, clusters = centers.shape
    output = np.zeros((genes, n_cells))
    if cluster_probs is None:
        cluster_probs = np.ones(clusters)/clusters
    for i in range(n_cells):
        c = np.random.choice(range(clusters), p=cluster_probs)
        output[:,i] = np.random.poisson(centers[:,c])
    return output

def generate_state_data(means, weights):
    """
    Generates data according to the Poisson Convex Mixture Model.

    Args:
        means (array): Cell types- genes x clusters
        weights (array): Cell cluster assignments- clusters x cells
    """
    x_true = np.dot(means, weights)
    sample = np.random.poisson(x_true)
    return sample.astype(float)
