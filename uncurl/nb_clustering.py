# Negative binomial clustering

import numpy as np
from scipy.optimize import fsolve, minimize
from scipy.special import gammaln, digamma, xlog1py

from .clustering import kmeans_pp
from . import pois_ll

eps=1e-8

def find_nb_genes(data):
    """
    Finds the indices of all genes in the dataset that have
    a mean < 0.9 variance. Returns an array of booleans.
    """
    data_means = data.mean(1)
    data_vars = data.var(1)
    nb_indices = data_means < 0.9*data_vars
    return nb_indices

def log_ncr(a, b):
    """
    Returns log(nCr(a,b)), given that b<a. Does not assume that a and b
    are integers (uses log-gamma).
    """
    val = gammaln(a+1) - gammaln(a-b+1) - gammaln(b+1)
    return val

def dlog_ncr(a, b):
    """
    derivative of log(nCr(a,b)) wrt a
    """
    return digamma(a+1) - digamma(a-b+1)

def nb_ll(data, P, R):
    """
    Returns the negative binomial log-likelihood of the data.

    Args:
        data (array): genes x cells
        P (array): NB success probability param - genes x clusters
        R (array): NB stopping param - genes x clusters

    Returns:
        cells x clusters array of log-likelihoods
    """
    # TODO: include factorial...
    #data = data + eps
    genes, cells = data.shape
    clusters = P.shape[1]
    lls = np.zeros((cells, clusters))
    for c in range(clusters):
        P_c = P[:,c].reshape((genes, 1))
        R_c = R[:,c].reshape((genes, 1))
        # don't need constant factors...
        ll = gammaln(R_c + data) - gammaln(R_c) #- gammaln(data + 1)
        ll += data*np.log(P_c) + xlog1py(R_c, -P_c)
        #new_ll = np.sum(nbinom.logpmf(data, R_c, P_c), 0)
        lls[:,c] = ll.sum(0)
    return lls

def zinb_ll(data, P, R, Z):
    """
    Returns the zero-inflated negative binomial log-likelihood of the data.
    """
    lls = nb_ll(data, P, R)
    clusters = P.shape[1]
    for c in range(clusters):
        pass
    return lls

def nb_ll_row(params, data_row):
    """
    returns the negative LL of a single row.

    Args:
        params (array) - [p, r]
        data_row (array) - 1d array of data

    Returns:
        LL of row
    """
    p = params[0]
    r = params[1]
    n = len(data_row)
    ll = np.sum(gammaln(data_row + r)) - np.sum(gammaln(data_row + 1))
    ll -= n*gammaln(r)
    ll += np.sum(data_row)*np.log(p)
    ll += n*r*np.log(1-p)
    return -ll

def nb_r_deriv(r, data_row):
    """
    Derivative of log-likelihood wrt r (formula from wikipedia)

    Args:
        r (float): the R paramemter in the NB distribution
        data_row (array): 1d array of length cells
    """
    n = len(data_row)
    d = sum(digamma(data_row + r)) - n*digamma(r) + n*np.log(r/(r+np.mean(data_row)))
    return d

def nb_fit(data, P_init=None, R_init=None, epsilon=1e-8, max_iters=100):
    """
    Fits the NB distribution to data using method of moments.

    Args:
        data (array): genes x cells
        P_init (array, optional): NB success prob param - genes x 1
        R_init (array, optional): NB stopping param - genes x 1

    Returns:
        P, R - fit to data
    """
    means = data.mean(1)
    variances = data.var(1)
    if (means > variances).any():
        raise ValueError("For NB fit, means must be less than variances")
    genes, cells = data.shape
    # method of moments
    P = 1.0 - means/variances
    R = means*(1-P)/P
    for i in range(genes):
        result = minimize(nb_ll_row, [P[i], R[i]], args=(data[i,:],),
                bounds = [(0, 1), (eps, None)])
        params = result.x
        P[i] = params[0]
        R[i] = params[1]
        #R[i] = fsolve(nb_r_deriv, R[i], args = (data[i,:],))
        #P[i] = data[i,:].mean()/(data[i,:].mean() + R[i])
    return P,R

def zinb_ll_row(params, data_row):
    """
    For use with optimization - returns ZINB parameters for a given row
    """
    # TODO

def nb_cluster(data, k, P_init=None, R_init=None, assignments=None, means=None, max_iters=10):
    """
    Performs negative binomial clustering on the given data. If some genes have mean > variance, then these genes are fitted to a Poisson distribution.

    Args:
        data (array): genes x cells
        k (int): number of clusters
        P_init (array): NB success prob param - genes x k. Default: random
        R_init (array): NB stopping param - genes x k. Default: random
        assignments (array): cells x 1 array of integers 0...k-1. Default: kmeans-pp (poisson)
        means (array): initial cluster means (for use with kmeans-pp to create initial assignments). Default: None
        max_iters (int): default: 100

    Returns:
        assignments (array): 1d array of length cells, containing integers 0...k-1
        P (array): genes x k - value is 0 for genes with mean > var
        R (array): genes x k - value is inf for genes with mean > var
    """
    genes, cells = data.shape
    if P_init is None:
        P_init = np.random.random((genes, k))
    if R_init is None:
        R_init = np.random.randint(1, data.max(), (genes, k))
        R_init = R_init.astype(float)
    if assignments is None:
        _, assignments = kmeans_pp(data, k, means)
    means = np.zeros((genes, k))
        #assignments = np.array([np.random.randint(0,k) for i in range(cells)])
    old_assignments = np.copy(assignments)
    # If mean > variance, then fall back to Poisson, since NB
    # distribution can't handle that case.
    for i in range(max_iters):
        # estimate params from assigned cells
        nb_gene_indices = fit_cluster(data, assignments, k, P_init, R_init, means)
        # re-calculate assignments
        lls = nb_ll(data[nb_gene_indices, :], P_init[nb_gene_indices,:], R_init[nb_gene_indices,:])
        lls += pois_ll.poisson_ll(data[~nb_gene_indices,:], means[~nb_gene_indices,:])
        # set NB params to failure values
        P_init[~nb_gene_indices,:] = 0
        R_init[~nb_gene_indices,:] = np.inf
        for c in range(cells):
            assignments[c] = np.argmax(lls[c,:])
        if np.equal(assignments,old_assignments).all():
            break
        old_assignments = np.copy(assignments)
    return assignments, P_init, R_init

def fit_cluster(data, assignments, k, P_init, R_init, means):
    """
    Fits NB/poisson params to a cluster.
    """
    for c in range(k):
        if data[:,assignments==c].shape[1] == 0:
            _, assignments = kmeans_pp(data, k)
    genes, cells = data.shape
    nb_gene_indices = np.array([True for i in range(genes)])
    for c in range(k):
        c_data = data[:,assignments==c]
        nb_gene_indices = nb_gene_indices & find_nb_genes(c_data)
    for c in range(k):
        c_data = data[:,assignments==c]
        nb_genes = c_data[nb_gene_indices,:]
        poisson_genes = c_data[~nb_gene_indices, :]
        P_init[nb_gene_indices, c], R_init[nb_gene_indices, c] = nb_fit(nb_genes)
        means[~nb_gene_indices, c] = poisson_genes.mean(1)
    return nb_gene_indices
