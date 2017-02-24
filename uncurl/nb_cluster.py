# Negative binomial clustering

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, digamma

eps=1e-4

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
        P (array): NB failure probability param - genes x clusters
        R (array): NB stopping param - genes x clusters

    Returns:
        cells x clusters array of log-likelihoods
    """
    data = data + eps
    genes, cells = data.shape
    clusters = P.shape[1]
    lls = np.zeros((cells, clusters))
    for c in range(clusters):
        P_c = P[:,c].reshape((genes, 1))
        R_c = R[:,c].reshape((genes, 1))
        new_ll = np.sum(log_ncr(data + R_c - 1, data) + R_c*np.log(1 - P_c) + data*np.log(P_c), 0)
        lls[:,c] = new_ll
    return lls

def _r_ll(R, P, data):
    """
    Returns negative log-likelihood
    """
    data = data + eps
    genes, cells = data.shape
    P_c = P.reshape((genes, 1))
    R_c = R.reshape((genes, 1))
    new_ll = np.sum(log_ncr(data + R_c - 1, data) + R_c*np.log(1 - P_c) + data*np.log(P_c))
    return -new_ll

def _r_deriv(R, P, data):
    """
    Gradient of LL w/r/t R

    Output is of shape genes x 1
    """
    genes, cells = data.shape
    P_ = P.reshape((genes, 1))
    R_ = R.reshape((genes, 1))
    d = np.sum(dlog_ncr(data + R_-1, data) + np.log(1-P_), 1)
    return -d

def nb_fit(data, P_init=None, R_init=None, epsilon=1e-8, maxiters=100):
    """
    Fits the NB distribution to data...

    Args:
        data (array): genes x cells
        P (array): NB failure prob param - genes x 1
        R (array): NB stopping param - genes x 1

    Returns:
        P, R - fit to data
    """
    # TODO
    genes, cells = data.shape
    data += eps
    if P_init is None:
        P_init = np.random.random(genes)
    if R_init is None:
        R_init = np.zeros(genes)
        for i in range(genes):
            R_init[i] = np.random.randint(1, 2*int(data[i,:].max()))
    e = np.inf
    R_init = R_init.astype(float)
    # successive minimization
    p_bounds = [(0,1.0) for i in range(genes)]
    r_bounds = [(0,None) for i in range(genes)]
    for i in range(maxiters):
        # minimize LL wrt P - this can be computed analytically
        ds = np.sum(data, 1)
        P_init = ds/(cells*R_init + ds)
        # minimize LL wrt R
        result = minimize(_r_ll, R_init, (P_init, data), jac=_r_deriv,
                bounds=r_bounds)
        R_init = result.x
        if np.abs(result.fun-e) <= epsilon:
            break
        e = result.fun
    return P_init, R_init

def nb_cluster(data, P_init=None, R_init=None):
    """
    """
    # TODO
