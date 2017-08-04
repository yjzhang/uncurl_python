# state estimation with poisson convex mixture model

from clustering import kmeans_pp

import numpy as np
from scipy.optimize import minimize

eps=1e-8

def _create_w_objective(m, X):
    """
    Creates an objective function and its derivative for W, given M and X (data)

    Args:
        m (array): genes x clusters
        X (array): genes x cells
    """
    genes, clusters = m.shape
    cells = X.shape[1]
    m_sum = m.sum(0)
    def objective(w):
        # convert w into a matrix first... because it's a vector for
        # optimization purposes
        w = w.reshape((m.shape[1], X.shape[1]))
        d = m.dot(w)+eps
        # derivative of objective wrt all elements of w
        # for w_{ij}, the derivative is... m_j1+...+m_jn sum over genes minus 
        # x_ij
        temp = X/d
        m2 = m.T.dot(temp)
        deriv = m_sum.reshape((clusters, 1)) - m2
        return np.sum(d - X*np.log(d))/genes, deriv.flatten()/genes
    return objective

def _create_m_objective(w, X):
    """
    Creates an objective function and its derivative for M, given W and X

    Args:
        w (array): clusters x cells
        X (array): genes x cells
    """
    clusters, cells = w.shape
    genes = X.shape[0]
    w_sum = w.sum(1)
    def objective(m):
        m = m.reshape((X.shape[0], w.shape[0]))
        d = m.dot(w)+eps
        temp = X/d
        w2 = w.dot(temp.T)
        deriv = w_sum - w2.T
        return np.sum(d - X*np.log(d))/genes, deriv.flatten()/genes
    return objective

# TODO: nolips

def nolips_update_w(X, M, W, disp=False):
    """
    Iteratively runs nolips updates.
    """
    genes, cells = X.shape
    k = W.shape[0]
    MW = M.dot(W)
    W_new = np.copy(W)
    R = M.sum(0)
    for i in range(cells):
        if disp:
            print('cell: {0}'.format(i))
        eta = 1./(2*X[:,i].sum())
        w = W[:,i]
        x = np.tile(X[:,i], (k,1)).T
        mw = np.tile(MW[:,i], (k,1)).T
        y2 = M*x
        C = np.sum(y2/mw, 0)
        p = w/(1 + eta*w*(R - C))
        W_new[:,i] = np.max(np.array([np.zeros(k), p]), 0)
    return W_new

def initialize_from_assignments(assignments, k, max_assign_weight=0.75):
    """
    Creates a weight initialization matrix from Poisson clustering assignments.

    Args:
        assignments (array): 1D array of integers, of length cells
        k (int): number of states/clusters
        max_assign_weight (float, optional): between 0 and 1 - how much weight to assign to the highest cluster. Default: 0.75

    Returns:
        init_W (array): k x cells
    """
    cells = len(assignments)
    init_W = np.zeros((k, cells))
    for i, a in enumerate(assignments):
        # entirely arbitrary... maybe it would be better to scale
        # the weights based on k?
        init_W[a, i] = max_assign_weight
        for a2 in range(k):
            if a2!=a:
                init_W[a2, i] = max_assign_weight/(k-1)
    return init_W

# TODO: add reps - number of starting points
def poisson_estimate_state(data, clusters, init_means=None, init_weights=None, max_iters=10, tol=1e-4, disp=True, inner_max_iters=25, reps=1, normalize=True):
    """
    Uses a Poisson Covex Mixture model to estimate cell states and
    cell state mixing weights.

    Args:
        data (array): genes x cells
        clusters (int): number of mixture components
        init_means (array, optional): initial centers - genes x clusters. Default: kmeans++ initializations
        init_weights (array, optional): initial weights - clusters x cells, or assignments as produced by clustering. Default: random(0,1)
        max_iters (int, optional): maximum number of iterations. Default: 10
        tol (float, optional): if both M and W change by less than tol, then the iteration is stopped. Default: 1e-4
        disp (bool, optional): whether or not to display optimization parameters. Default: True
        inner_max_iters (int, optional): Number of iterations to run in the scipy minimizer for M and W. Default: 400
        reps (int, optional): number of random initializations. Default: 1.
        normalize (bool, optional): True if the resulting W should sum to 1 for each cell. Default: True.

    Returns:
        M (array): genes x clusters - state means
        W (array): clusters x cells - state mixing components for each cell
        ll (float): final log-likelihood
    """
    genes, cells = data.shape
    if init_means is None:
        means, assignments = kmeans_pp(data, clusters)
    else:
        means = init_means.copy()
    clusters = means.shape[1]
    w_init = np.random.random((clusters,cells))
    if init_weights is not None:
        if len(init_weights.shape)==1:
            init_weights = initialize_from_assignments(init_weights, clusters)
        w_init = init_weights.copy()
    m_init = means.reshape(genes*clusters)
    # repeat steps 1 and 2 until convergence:
    ll = np.inf
    # arbitrary argument
    nolips_iters = inner_max_iters
    for i in range(max_iters):
        if disp:
            print('iter: {0}'.format(i))
        # step 1: given M, estimate W
        w_objective = _create_w_objective(means, data)
        # TODO: select between L-BFGS-B or SLSQP optimization methods
        #w_res = minimize(w_objective, w_init, method='L-BFGS-B', jac=True, bounds=w_bounds, options={'disp':disp, 'maxiter':inner_max_iters})
        #w_diff = np.sqrt(np.sum((w_res.x-w_init)**2))/w_init.size
        for i in range(nolips_iters):
            w_new = nolips_update_w(data+eps, means, w_init)
            #w_new = w_res.x.reshape((clusters, cells))
            #w_new = w_new/w_new.sum(0)
            w_init = w_new
        w_ll, w_deriv = w_objective(w_new.reshape(clusters*cells))
        if disp:
            print('Finished updating W. Objective value: {0}'.format(w_ll))
        # step 2: given W, update M
        for i in range(nolips_iters):
            m_new = nolips_update_w(data.T+eps, w_new.T, means.T)
            means = m_new.T
        m_objective = _create_m_objective(w_new, data)
        m_ll, m_deriv = m_objective(means.reshape(genes*clusters))
        if disp:
            print('Finished updating M. Objective value: {0}'.format(m_ll))
        # method could be 'L-BFGS-B' or 'SLSQP'... SLSQP gives a memory error...
        # or use TNC...
        #m_res = minimize(m_objective, m_init, method='L-BFGS-B', jac=True, bounds=m_bounds, options={'disp':disp, 'maxiter':inner_max_iters})
        #ll = m_res.fun
        #m_diff = np.sqrt(np.sum((m_res.x-m_init)**2))/m_init.size
        #if w_diff < tol and m_diff < tol:
        #    break
    if normalize:
        w_new = w_new/w_new.sum(0)
    return means, w_new, ll
