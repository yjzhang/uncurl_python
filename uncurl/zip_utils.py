import numpy as np
from scipy import sparse
from scipy.stats import poisson
from scipy.special import xlogy, gammaln

eps = 1e-10


def zip_ll(data, means, M):
    """
    Calculates the zero-inflated Poisson log-likelihood.

    Args:
        data (array): genes x cells
        means (array): genes x k
        M (array): genes x k - this is the zero-inflation parameter.

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
        ll_0 = np.where((L_i==0) & (means_i==0), -means_i, ll_0)
        # not including constant factors
        ll_1 = np.log(1 - L_i) + xlogy(data, means_i) -  means_i
        ll_0 = np.where(d0, ll_0, 0.0)
        ll_1 = np.where(d1, ll_1, 0.0)
        ll[:,i] = np.sum(ll_0 + ll_1, 0)
    return ll

def zip_ll_row(params, data_row):
    """
    Returns the negative log-likelihood of a row given ZIP data.

    Args:
        params (list): [lambda zero-inf]
        data_row (array): 1d array

    Returns:
        negative log-likelihood
    """
    l = params[0]
    pi = params[1]
    d0 = (data_row==0)
    likelihood = d0*pi + (1-pi)*poisson.pmf(data_row, l)
    return -np.log(likelihood+eps).sum()

