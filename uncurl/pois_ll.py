# Poisson log-likelihood

import numpy as np
from scipy import sparse
from scipy.special import xlogy, gammaln

from uncurl.sparse_utils import sparse_poisson_ll_csc

eps = 1e-10

def sparse_poisson_ll(data, means):
    data = sparse.csc_matrix(data)
    return sparse_poisson_ll_csc(
            data.data,
            data.indices,
            data.indptr,
            data.shape[0],
            data.shape[1],
            means,
            eps)

def poisson_ll(data, means):
    """
    Calculates the Poisson log-likelihood.

    Args:
        data (array): 2d numpy array of genes x cells
        means (array): 2d numpy array of genes x k

    Returns:
        cells x k array of log-likelihood for each cell/cluster pair
    """
    if sparse.issparse(data):
        return sparse_poisson_ll(data, means)
    genes, cells = data.shape
    clusters = means.shape[1]
    ll = np.zeros((cells, clusters))
    for i in range(clusters):
        means_i = np.tile(means[:,i], (cells, 1))
        means_i = means_i.transpose() + eps
        #ll[:,i] = np.sum(xlogy(data, means_i) - gammaln(data+1) - means_i, 0)
        ll[:,i] = np.sum(xlogy(data, means_i) - means_i, 0)
    return ll

def poisson_ll_2(p1, p2):
    """
    Calculates Poisson LL(p1|p2).
    """
    p1_1 = p1 + eps
    p2_1 = p2 + eps
    return np.sum(-p2_1 + p1_1*np.log(p2_1))

def poisson_dist(p1, p2):
    """
    Calculates the Poisson distance between two vectors.

    p1 can be a sparse matrix, while p2 has to be a dense matrix.
    """
    # ugh...
    p1_ = p1 + eps
    p2_ = p2 + eps
    return np.dot(p1_-p2_, np.log(p1_/p2_))

