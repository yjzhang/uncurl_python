# dimensionality reduction

import numpy as np
from .pois_ll import poisson_dist

eps=1e-8
max_or_zero = np.vectorize(lambda x: max(0.0,x))

def dim_reduce(means, weights, d):
    """
    Dimensionality reduction using Poisson distances and MDS.

    Args:
        means (array): genes x clusters
        weights (array): clusters x cells
        d (int): desired dimensionality

    Returns:
        X, a clusters x d matrix representing the reduced dimensions
        of the cluster centers.
    """
    return dim_reduce_data(means, d)

def dim_reduce_data(data, d):
    """
    Does a MDS on the data directly, not on the means.

    Args:
        data (array): genes x cells
        d (int): desired dimensionality

    Returns:
        X, a cells x d matrix
    """
    genes, cells = data.shape
    distances = np.zeros((cells, cells))
    for i in range(cells):
        for j in range(cells):
            distances[i,j] = poisson_dist(data[:,i], data[:,j])
    # do MDS on the distance matrix (procedure from Wikipedia)
    proximity = distances**2
    J = np.eye(cells) - 1./cells
    B = -0.5*np.dot(J, np.dot(proximity, J))
    # B should be symmetric, so we can use eigh
    e_val, e_vec = np.linalg.eigh(B)
    # Note: lam should be ordered to be the largest eigenvalues
    lam = np.diag(e_val[-d:])[::-1]
    #lam = max_or_zero(lam)
    E = e_vec[:,-d:][::-1]
    X = np.dot(E, lam**0.5)
    return X
