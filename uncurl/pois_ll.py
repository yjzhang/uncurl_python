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
            ll[k,i] = -sum(means[:,i] + data[:,k]*np.log(means[:,i]))
    return ll
