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
        output - array with shape genes x n_cells
        labels - array of cluster labels
    """
    genes, clusters = centers.shape
    output = np.zeros((genes, n_cells))
    if cluster_probs is None:
        cluster_probs = np.ones(clusters)/clusters
    labels = []
    for i in range(n_cells):
        c = np.random.choice(range(clusters), p=cluster_probs)
        labels.append(c)
        output[:,i] = np.random.poisson(centers[:,c])
    return output, np.array(labels)

def generate_zip_data(M, L, n_cells, cluster_probs=None):
    """
    Generates zero-inflated poisson-distributed data, given a set of means and zero probs for each cluster.

    Args:
        M (array): genes x clusters matrix
        L (array): genes x clusters matrix - zero-inflation parameters
        n_cells (int): number of output cells
        cluster_probs (array): prior probability for each cluster.
            Default: uniform.

    Returns:
        output - array with shape genes x n_cells
        labels - array of cluster labels
    """
    genes, clusters = M.shape
    output = np.zeros((genes, n_cells))
    if cluster_probs is None:
        cluster_probs = np.ones(clusters)/clusters
    zip_p = np.random.random((genes, n_cells))
    labels = []
    for i in range(n_cells):
        c = np.random.choice(range(clusters), p=cluster_probs)
        labels.append(c)
        output[:,i] = np.where(zip_p[:,i] < L[:,c], 0, np.random.poisson(M[:,c]))
    return output, np.array(labels)


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

def generate_zip_state_data(means, weights, z):
    """
    Generates data according to the Zero-inflated Poisson Convex Mixture Model.

    Args:
        means (array): Cell types- genes x clusters
        weights (array): Cell cluster assignments- clusters x cells
        z (float): zero-inflation parameter

    Returns:
        data matrix - genes x cells
    """
    x_true = np.dot(means, weights)
    sample = np.random.poisson(x_true)
    random = np.random.random(x_true.shape)
    x_true[random < z] = 0
    return sample.astype(float)

def generate_nb_state_data(means, weights, R):
    """
    Generates data according to the Negative Binomial Convex Mixture Model.

    Args:
        means (array): Cell types- genes x clusters
        weights (array): Cell cluster assignments- clusters x cells
        R (array): dispersion parameter - 1 x genes

    Returns:
        data matrix - genes x cells
    """
    cells = weights.shape[1]
    # x_true = true means
    x_true = np.dot(means, weights)
    # convert means into P
    R_ = np.tile(R, (cells, 1)).T
    P_true = x_true/(R_ + x_true)
    sample = np.random.negative_binomial(np.tile(R, (cells, 1)).T, P_true)
    return sample.astype(float)

def generate_nb_states(n_states, n_cells, n_genes):
    """
    Generates means and weights for the Negative Binomial Mixture Model.
    Weights are distributed Dirichlet(1,1,...), means are rand(0, 1).
    Returned values can be passed to generate_state_data(M, W).

    Args:
        n_states (int): number of states or clusters
        n_cells (int): number of cells
        n_genes (int): number of genes

    Returns:
        M - genes x clusters
        W - clusters x cells
        R - genes x 1 - randint(1, 100)
    """
    W = np.random.dirichlet([1]*n_states, size=(n_cells,))
    W = W.T
    M = np.random.random((n_genes, n_states))*100
    R = np.random.randint(1, 100, n_genes)
    return M, W, R

def generate_poisson_states(n_states, n_cells, n_genes):
    """
    Generates means and weights for the Poisson Convex Mixture Model.
    Weights are distributed Dirichlet(1,1,...), means are rand(0, 100).
    Returned values can be passed to generate_state_data(M, W).

    Args:
        n_states (int): number of states or clusters
        n_cells (int): number of cells
        n_genes (int): number of genes

    Returns:
        M - genes x clusters
        W - clusters x cells
    """
    W = np.random.dirichlet([1]*n_states, size=(n_cells,))
    W = W.T
    M = np.random.random((n_genes, n_states))*100
    return M, W

def generate_poisson_lineage(n_states, n_cells_per_cluster, n_genes, means=300):
    """
    Generates a lineage for each state- assumes that each state has a common
    ancestor.

    Returns:
        M - genes x clusters
        W - clusters x cells
    """
    # means...
    M = np.random.random((n_genes, n_states))*means
    center = M.mean(1)
    W = np.zeros((n_states, n_cells_per_cluster*n_states))
    # TODO
    # start at a center where all the clusters have equal probability, and for
    # each cluster, interpolate linearly towards the cluster.
    index = 0
    means = np.array([1.0/n_states]*n_states)
    for c in range(n_states):
        for i in range(n_cells_per_cluster):
            w = np.copy(means)
            new_value = w[c] + i*(1.0 - 1.0/n_states)/n_cells_per_cluster
            w[:] = (1.0 - new_value)/(n_states - 1.0)
            w[c] = new_value
            W[:, index] = w
            index += 1
    return M, W

def generate_nb_data(P, R, n_cells, assignments=None):
    """
    Generates negative binomial data

    Args:
        P (array): genes x clusters
        R (array): genes x clusters
        n_cells (int): number of cells
        assignments (list): cluster assignment of each cell. Default:
            random uniform

    Returns:
        data array with shape genes x cells
        labels - array of cluster labels
    """
    genes, clusters = P.shape
    output = np.zeros((genes, n_cells))
    if assignments is None:
        cluster_probs = np.ones(clusters)/clusters
    labels = []
    for i in range(n_cells):
        if assignments is None:
            c = np.random.choice(range(clusters), p=cluster_probs)
        else:
            c = assignments[i]
        labels.append(c)
        # because numpy's negative binomial, r is the number of successes
        output[:,i] = np.random.negative_binomial(R[:,c], 1.0-P[:,c])
    return output, np.array(labels)


