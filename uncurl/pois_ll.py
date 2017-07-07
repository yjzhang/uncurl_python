# Poisson log-likelihood

import numpy as np
from scipy import sparse
from scipy.stats import poisson
from scipy.special import xlogy, gammaln

eps = 1e-8

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
    p1_1 = p1 + eps
    p2_1 = p2 + eps
    return -np.sum(p2_1 + p1_1*np.log(p2_1))

def poisson_dist(p1, p2):
    """
    Calculates the Poisson distance between two vectors.
    """
    # ugh...
    p1_ = p1 + eps
    p2_ = p2 + eps
    return np.dot(p1_-p2_, np.log(p1_/p2_))

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
        ll_0 = np.log(L_i + (1 - L_i)*np.exp(-means_i))
        ll_0 = np.where(ll_0==-np.inf, -means_i, ll_0)
        ll_1 = np.log(1 - L_i) + xlogy(data, means_i) - gammaln(data+1) - means_i
        ll_0 = np.where(d0, ll_0, 0.0)
        ll_1 = np.where(d1, ll_1, 0.0)
        ll[:,i] = np.sum(ll_0 + ll_1, 0)
    return ll
