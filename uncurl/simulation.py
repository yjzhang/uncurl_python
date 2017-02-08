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

    Returns:
        data matrix - genes x cells
    """
    x_true = np.dot(means, weights)
    sample = np.random.poisson(x_true)
    return sample.astype(float)

def generate_poisson_states(n_states, n_cells, n_genes):
    """
    Generates means and weights for the Poisson Convex Mixture Model.
    Weights are distributed Dirichlet(1,1,...), means are rand(0, 100).
    Returned values can be passed to generate_state_data(M, W).

    Args:
        n_states (int) - number of states or clusters
        n_cells (int) - number of cells
        n_genes (int) - number of genes

    Returns:
        M - genes x clusters
        W - clusters x cells
    """
    W = np.random.dirichlet([1]*n_states, size=(n_states,))
    W = W.T
    M = np.random.random((n_genes, n_states))*100
    return M, W
