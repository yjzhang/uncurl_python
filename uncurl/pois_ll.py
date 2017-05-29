# Poisson log-likelihood

import numpy as np
from scipy.stats import poisson
from scipy.special import xlogy, gammaln

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
        ll[:,i] = np.sum(xlogy(data, means_i) - gammaln(data+1) - means_i, 0)
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

# TODO: poisson_ll_zip

def zip_ll(data, means, M):
    """
    Calculates the zero-inflated Poisson log-likelihood.

    Args:
        data (array): genes x cells
        means (array): genes x k
        M (array): genes x k - this is the probability of having a count of zero.

    Returns:
        cells x k array of log-likelihood for each cell/cluster pair.
    """
    genes, cells = data.shape
    clusters = means.shape[1]
    ll = np.zeros((cells, clusters))
    d0 = (data==0)
    d1 = (data>0)
    for i in range(clusters):
        means_i = np.tile(means[:,i], (cells, 1))
        means_i = means_i.transpose()
        L_i = np.tile(M[:,i], (cells, 1))
        L_i = L_i.transpose()
        ll[:,i] = np.sum(d0*np.log(L_i + (1 - L_i)*np.exp(-means_i)), 0)
        ll[:,i] += np.sum(d1*(np.log(1 - L_i) + xlogy(data, means_i) - gammaln(data+1) - means_i), 0)
    return ll
