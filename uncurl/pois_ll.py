# Poisson log-likelihood

import numpy as np

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
        for k  in range(cells):
            ll[k,i] = poisson_ll_2(data[:,k], means[:,i])
    return ll

def poisson_ll_2(p1, p2):
    """
    Calculates Poisson LL(p1|p2).
    """
    return -sum(p2 + p1*np.log(p2))

def poisson_dist(p1, p2):
    """
    Calculates the Poisson distance between two vectors.
    """
    return np.dot(p1-p2, np.log(p1/p2))
