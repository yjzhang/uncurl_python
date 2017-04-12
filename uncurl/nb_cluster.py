# Negative binomial clustering

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, digamma, xlog1py
from scipy.stats import nbinom

from poisson_cluster import kmeans_pp
import pois_ll

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
    #data = data + eps
    genes, cells = data.shape
    clusters = P.shape[1]
    lls = np.zeros((cells, clusters))
    for c in range(clusters):
        P_c = P[:,c].reshape((genes, 1))
        R_c = R[:,c].reshape((genes, 1))
        ll = gammaln(R_c + data) - gammaln(data + 1) - gammaln(R_c)
        ll += R_c*np.log(P_c) + xlog1py(data, -P_c)
        #new_ll = np.sum(nbinom.logpmf(data, R_c, P_c), 0)
        lls[:,c] = ll.sum(0)
    return lls

def nb_fit(data, P_init=None, R_init=None, epsilon=1e-8, max_iters=100):
    """
    Fits the NB distribution to data using method of moments.

    Args:
        data (array): genes x cells
        P (array): NB success prob param - genes x 1
        R (array): NB stopping param - genes x 1

    Returns:
        P, R - fit to data
    """
    # method of moments
    means = data.mean(1)
    variances = data.var(1)
    if (means > variances).any():
        raise ValueError("For NB fit, means must be less than variances")
    P = 1.0 - means/variances
    R = means*(1-P)/P
    return P,R

def nb_cluster(data, k, P_init=None, R_init=None, assignments=None, max_iters=10):
    """
    Performs negative binomial clustering on the given data. If some genes have mean > variance, then these genes are fitted to a Poisson distribution.

    Args:
        data (array): genes x cells
        k (int): number of clusters
        P_init (array): NB success prob param - genes x k. Default: random
        R_init (array): NB stopping param - genes x k. Default: random
        assignments (array): cells x 1 array of integers 0...k-1. Default: kmeans-pp (poisson)
        max_iters (int): default: 100

    Returns:
        P (array): genes x k - value is 0 for genes with mean > var
        R (array): genes x k - value is inf for genes with mean > var
        assignments (array): 1d array of length cells, containing integers 0...k-1
    """
    genes, cells = data.shape
    if P_init is None:
        P_init = np.random.random((genes, k))
    if R_init is None:
        R_init = np.random.randint(1, data.max(), (genes, k))
    if assignments is None:
        _, assignments = kmeans_pp(data, k)
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
    return P_init, R_init, assignments

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
