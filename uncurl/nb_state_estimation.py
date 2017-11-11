# state estimation with NB convex mixture model

from .clustering import kmeans_pp
from .nb_clustering import nb_fit, find_nb_genes
from .state_estimation import initialize_from_assignments

import numpy as np
from scipy.optimize import minimize

eps=1e-8

def _create_w_objective(m, X, R):
    """
    Creates an objective function and its derivative for W, given M and X (data)

    Args:
        m (array): genes x clusters
        X (array): genes x cells
        R (array): 1 x genes
    """
    genes, clusters = m.shape
    cells = X.shape[1]
    R1 = R.reshape((genes, 1)).dot(np.ones((1, cells)))
    def objective(w):
        # convert w into a matrix first... because it's a vector for
        # optimization purposes
        w = w.reshape((m.shape[1], X.shape[1]))
        d = m.dot(w)+eps
        return np.sum((X + R1)*np.log(d + R1) - X*np.log(d))/genes
    def deriv(w):
        # derivative of objective wrt all elements of w
        # for w_{ij}, the derivative is... m_j1+...+m_jn sum over genes minus 
        # x_ij
        w2 = w.reshape((m.shape[1], X.shape[1]))
        d = m.dot(w2)+eps
        temp = X/d
        temp2 = (X+R1)/(d+R1)
        m1 = m.T.dot(temp2)
        m2 = m.T.dot(temp)
        deriv = m1 - m2
        return deriv.flatten()/genes
    return objective, deriv

def _create_m_objective(w, X, R):
    """
    Creates an objective function and its derivative for M, given W and X

    Args:
        w (array): clusters x cells
        X (array): genes x cells
        R (array): 1 x genes
    """
    clusters, cells = w.shape
    genes = X.shape[0]
    R1 = R.reshape((genes, 1)).dot(np.ones((1, cells)))
    def objective(m):
        m = m.reshape((X.shape[0], w.shape[0]))
        d = m.dot(w)+eps
        return np.sum((X+R1)*np.log(d + R1) - X*np.log(d))/genes
    def deriv(m):
        m2 = m.reshape((X.shape[0], w.shape[0]))
        d = m2.dot(w)+eps
        temp = X/d
        temp2 = (X+R1)/(d+R1)
        w1 = w.dot(temp2.T)
        w2 = w.dot(temp.T)
        deriv = w1.T - w2.T
        return deriv.flatten()/genes
    return objective, deriv

def nb_estimate_state(data, clusters, R=None, init_means=None, init_weights=None, max_iters=10, tol=1e-4, disp=True, inner_max_iters=400, normalize=True):
    """
    Uses a Negative Binomial Mixture model to estimate cell states and
    cell state mixing weights.

    If some of the genes do not fit a negative binomial distribution
    (mean > var), then the genes are discarded from the analysis.

    Args:
        data (array): genes x cells
        clusters (int): number of mixture components
        R (array, optional): vector of length genes containing the dispersion estimates for each gene. Default: use nb_fit
        init_means (array, optional): initial centers - genes x clusters. Default: kmeans++ initializations
        init_weights (array, optional): initial weights - clusters x cells. Default: random(0,1)
        max_iters (int, optional): maximum number of iterations. Default: 10
        tol (float, optional): if both M and W change by less than tol (in RMSE), then the iteration is stopped. Default: 1e-4
        disp (bool, optional): whether or not to display optimization parameters. Default: True
        inner_max_iters (int, optional): Number of iterations to run in the scipy minimizer for M and W. Default: 400
        normalize (bool, optional): True if the resulting W should sum to 1 for each cell. Default: True.

    Returns:
        M (array): genes x clusters - state centers
        W (array): clusters x cells - state mixing components for each cell
        R (array): 1 x genes - NB dispersion parameter for each gene
        ll (float): Log-likelihood of final iteration
    """
    # TODO: deal with non-NB data... just ignore it? or do something else?
    data_subset = data.copy()
    genes, cells = data_subset.shape
    # 1. use nb_fit to get inital Rs
    if R is None:
        nb_indices = find_nb_genes(data)
        data_subset = data[nb_indices, :]
        if init_means is not None and len(init_means) > sum(nb_indices):
            init_means = init_means[nb_indices, :]
        genes, cells = data_subset.shape
        R = np.zeros(genes)
        P, R = nb_fit(data_subset)
    if init_means is None:
        means, assignments = kmeans_pp(data_subset, clusters)
    else:
        means = init_means.copy()
    clusters = means.shape[1]
    w_init = np.random.random(cells*clusters)
    if init_weights is not None:
        if len(init_weights.shape)==1:
            init_weights = initialize_from_assignments(init_weights, clusters)
        w_init = init_weights.reshape(cells*clusters)
    m_init = means.reshape(genes*clusters)
    ll = np.inf
    # repeat steps 1 and 2 until convergence:
    for i in range(max_iters):
        if disp:
            print('iter: {0}'.format(i))
        w_bounds = [(0, 1.0) for x in w_init]
        m_bounds = [(0, None) for x in m_init]
        # step 1: given M, estimate W
        w_objective, w_deriv = _create_w_objective(means, data_subset, R)
        w_res = minimize(w_objective, w_init, method='L-BFGS-B', jac=w_deriv, bounds=w_bounds, options={'disp':disp, 'maxiter':inner_max_iters})
        w_diff = np.sqrt(np.sum((w_res.x-w_init)**2))/w_init.size
        w_new = w_res.x.reshape((clusters, cells))
        w_init = w_res.x
        # step 2: given W, update M
        m_objective, m_deriv = _create_m_objective(w_new, data_subset, R)
        # method could be 'L-BFGS-B' or 'SLSQP'... SLSQP gives a memory error...
        # or use TNC...
        m_res = minimize(m_objective, m_init, method='L-BFGS-B', jac=m_deriv, bounds=m_bounds, options={'disp':disp, 'maxiter':inner_max_iters})
        m_diff = np.sqrt(np.sum((m_res.x-m_init)**2))/m_init.size
        m_new = m_res.x.reshape((genes, clusters))
        m_init = m_res.x
        ll = m_res.fun
        means = m_new
        if w_diff < tol and m_diff < tol:
            break
    if normalize:
        w_new = w_new/w_new.sum(0)
    return m_new, w_new, R, ll
