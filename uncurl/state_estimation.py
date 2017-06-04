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
    def objective(w):
        # convert w into a matrix first... because it's a vector for
        # optimization purposes
        w = w.reshape((m.shape[1], X.shape[1]))
        d = m.dot(w)+eps
        # derivative of objective wrt all elements of w
        # for w_{ij}, the derivative is... m_j1+...+m_jn sum over genes minus 
        # x_ij
        temp = X/d
        m_sum = m.sum(0)
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
    def objective(m):
        m = m.reshape((X.shape[0], w.shape[0]))
        d = m.dot(w)+eps
        temp = X/d
        w_sum = w.sum(1)
        w2 = w.dot(temp.T)
        deriv = w_sum - w2.T
        return np.sum(d - X*np.log(d))/genes, deriv.flatten()/genes
    return objective

# TODO: add reps - number of starting points
def poisson_estimate_state(data, clusters, init_means=None, init_weights=None, max_iters=10, tol=1e-4, disp=True, inner_max_iters=400, reps=1):
    """
    Uses a Poisson Covex Mixture model to estimate cell states and
    cell state mixing weights.

    Args:
        data (array): genes x cells
        clusters (int): number of mixture components
        init_means (array, optional): initial centers - genes x clusters. Default: kmeans++ initializations
        init_weights (array, optional): initial weights - clusters x cells. Default: random(0,1)
        max_iters (int, optional): maximum number of iterations. Default: 10
        tol (float, optional): if both M and W change by less than tol, then the iteration is stopped. Default: 1e-4
        disp (bool, optional): whether or not to display optimization parameters. Default: True
        inner_max_iters (int, optional): Number of iterations to run in the scipy minimizer for M and W. Default: 400
        reps (int, optional): number of random initializations. Default: 1.

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
    w_init = np.random.random(cells*clusters)
    if init_weights is not None:
        w_init = init_weights.reshape(cells*clusters)
    m_init = means.reshape(genes*clusters)
    # repeat steps 1 and 2 until convergence:
    ll = np.inf
    for i in range(max_iters):
        if disp:
            print('iter: {0}'.format(i))
        w_bounds = [(0, None) for x in w_init]
        m_bounds = [(0, None) for x in m_init]
        # step 1: given M, estimate W
        w_objective = _create_w_objective(means, data)
        # TODO: select between L-BFGS-B or SLSQP optimization methods
        w_res = minimize(w_objective, w_init, method='L-BFGS-B', jac=True, bounds=w_bounds, options={'disp':disp, 'maxiter':inner_max_iters})
        w_diff = np.sqrt(np.sum((w_res.x-w_init)**2))/w_init.size
        w_new = w_res.x.reshape((clusters, cells))
        w_init = w_res.x
        # step 2: given W, update M
        m_objective = _create_m_objective(w_new, data)
        # method could be 'L-BFGS-B' or 'SLSQP'... SLSQP gives a memory error...
        # or use TNC...
        m_res = minimize(m_objective, m_init, method='L-BFGS-B', jac=True, bounds=m_bounds, options={'disp':disp, 'maxiter':inner_max_iters})
        ll = m_res.fun
        m_diff = np.sqrt(np.sum((m_res.x-m_init)**2))/m_init.size
        m_new = m_res.x.reshape((genes, clusters))
        m_init = m_res.x
        means = m_new
        if w_diff < tol and m_diff < tol:
            break
    w_new = w_new/w_new.sum(0)
    return m_new, w_new, ll
