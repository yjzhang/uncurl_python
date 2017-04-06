# Poisson log-likelihood

import numpy as np
from scipy.stats import poisson

def poisson_ll(data, means):
    """
    Calculates the Poisson log-likelihood.

    Args:
        data (array): 2d numpy array of genes x cells
        means (array): 2d numpy array of genes x k

    Returns:
        cells x k array of log-likelihood for each cell/cluster pair
    """
    genes, cells = data.shape
    clusters = means.shape[1]
    ll = np.zeros((cells, clusters))
    for i in range(clusters):
        means_i = np.tile(means[:,i], (cells, 1))
        means_i = means_i.transpose()
        ll[:,i] = np.sum(poisson.logpmf(data+1e-8, means_i+1e-8), 0)
    return ll

def poisson_ll_2(p1, p2):
    """
    Calculates Poisson LL(p1|p2).
    """
    p1_1 = p1 + 1e-10
    p2_1 = p2 + 1e-10
    return -np.sum(p2_1 + p1_1*np.log(p2_1))

def poisson_dist(p1, p2):
    """
    Calculates the Poisson distance between two vectors.
    """
    # ugh...
    p1 += 1e-10
    p2 += 1e-10
    return np.dot(p1-p2, np.log(p1/p2))
