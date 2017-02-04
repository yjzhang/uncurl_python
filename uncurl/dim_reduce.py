# dimensionality reduction

import numpy as np
from pois_ll import poisson_dist

def dim_reduce(data, means, weights, d):
    """
    Dimensionality reduction using Poisson distances and MDS.

    Args:
        data (array) - genes x cells
        means (array) - genes x clusters
        weights (array) - clusters x cells
        d (int) - desired dimensionality

    Returns:
        X, a clusters x d matrix representing the reduced dimensions
        of the cluster centers.
    """
    genes, cells = data.shape
    clusters = means.shape[1]
    distances = np.zeros((clusters, clusters))
    for i in range(clusters):
        for j in range(clusters):
            distances[i,j] = poisson_dist(means[:,i], means[:,j])
    # do MDS on the distance matrix (procedure from Wikipedia)
    proximity = distances**2
    J = np.eye(clusters) - 1./clusters
    B = -0.5*np.dot(J, np.dot(proximity, J))
    e_val, e_vec = np.linalg.eig(B)
    lam = np.diag(e_val[:d])
    E = e_vec[:,:d]
    X = np.dot(E, lam**0.5)
    return X
