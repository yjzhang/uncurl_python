# state estimation with poisson convex mixture model

from clustering import kmeans_pp, poisson_cluster
from nolips import nolips_update_w, sparse_nolips_update_w, objective, sparse_objective

import numpy as np
from scipy import sparse
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

def run_state_estimation(data, clusters, dist='Poiss', reps=1, init_means=None, init_weights=None, method='NoLips', max_iters=10, tol=1e-10, disp=True, inner_max_iters=100, normalize=True):
    """
    Runs state estimation for multiple initializations, returning the result with the highest log-likelihood.

    Args:
        data (array): genes x cells
        clusters (int): number of mixture components
        dist (str, optional): Distribution used in state estimation. Options: 'Poiss', 'NB', 'ZIP'.
        reps (int, optional): number of random initializations. Default: 10.
        init_means (array, optional): initial centers - genes x clusters. Default: from Poisson kmeans
        init_weights (array, optional): initial weights - clusters x cells, or assignments as produced by clustering. Default: from Poisson kmeans
        method (str, optional): optimization method. Current options are 'NoLips' and 'L-BFGS-B'. Default: 'NoLips'.
        max_iters (int, optional): maximum number of iterations. Default: 10
        tol (float, optional): if both M and W change by less than tol (RMSE), then the iteration is stopped. Default: 1e-10
        disp (bool, optional): whether or not to display optimization parameters. Default: True
        inner_max_iters (int, optional): Number of iterations to run in the optimization subroutine for M and W. Default: 100
        normalize (bool, optional): True if the resulting W should sum to 1 for each cell. Default: True.

    Returns:
        M (array): genes x clusters - state means
        W (array): clusters x cells - state mixing components for each cell
        ll (float): final log-likelihood
    """
    # TODO: add reps - number of starting points
    func = poisson_estimate_state
    if dist=='Poiss':
        pass
    elif dist=='NB':
        pass
    elif dist=='ZIP':
        pass
    else:
        print('dist should be one of Poiss, NB, or ZIP. Using Poiss.')
    for i in range(reps):
        pass


def poisson_estimate_state(data, clusters, init_means=None, init_weights=None, method='NoLips', max_iters=10, tol=1e-10, disp=True, inner_max_iters=100, normalize=True):
    """
    Uses a Poisson Covex Mixture model to estimate cell states and
    cell state mixing weights.

    Args:
        data (array): genes x cells
        clusters (int): number of mixture components
        init_means (array, optional): initial centers - genes x clusters. Default: from Poisson kmeans
        init_weights (array, optional): initial weights - clusters x cells, or assignments as produced by clustering. Default: from Poisson kmeans
        method (str, optional): optimization method. Current options are 'NoLips' and 'L-BFGS-B'. Default: 'NoLips'.
        max_iters (int, optional): maximum number of iterations. Default: 10
        tol (float, optional): if both M and W change by less than tol (RMSE), then the iteration is stopped. Default: 1e-10
        disp (bool, optional): whether or not to display optimization parameters. Default: True
        inner_max_iters (int, optional): Number of iterations to run in the optimization subroutine for M and W. Default: 100
        normalize (bool, optional): True if the resulting W should sum to 1 for each cell. Default: True.

    Returns:
        M (array): genes x clusters - state means
        W (array): clusters x cells - state mixing components for each cell
        ll (float): final log-likelihood
    """
    genes, cells = data.shape
    if init_means is None:
        assignments, means = poisson_cluster(data, clusters)
        if init_weights is None:
            init_weights = initialize_from_assignments(assignments, clusters)
    else:
        means = init_means.copy()
    means = means.astype(float)
    if init_weights is not None:
        if len(init_weights.shape)==1:
            init_weights = initialize_from_assignments(init_weights, clusters)
        w_init = init_weights.copy()
    else:
        w_init = np.random.random((clusters, cells))
    # repeat steps 1 and 2 until convergence:
    nolips_iters = inner_max_iters
    X = data.astype(float)
    if sparse.issparse(X):
        update_fn = sparse_nolips_update_w
        Xsum = np.asarray(X.sum(0)).flatten()
        Xsum_m = np.asarray(X.sum(1)).flatten()
        # L-BFGS-B won't work right now for sparse matrices
        X = sparse.coo_matrix(X)
        method = 'NoLips'
        objective_fn = sparse_objective
    else:
        objective_fn = objective
        update_fn = nolips_update_w
        Xsum = X.sum(0)
        Xsum_m = X.sum(1)
    for i in range(max_iters):
        if disp:
            print('iter: {0}'.format(i))
        # step 1: given M, estimate W
        #w_diff = np.sqrt(np.sum((w_res.x-w_init)**2))/w_init.size
        if method=='NoLips':
            for j in range(nolips_iters):
                w_new = update_fn(X, means, w_init, Xsum)
                #w_new = w_res.x.reshape((clusters, cells))
                #w_new = w_new/w_new.sum(0)
                w_diff = np.sqrt(np.sum((w_new - w_init)**2)/(clusters*cells))
                w_init = w_new
                if w_diff < tol:
                    break
        elif method=='L-BFGS-B':
            w_objective = _create_w_objective(means, data)
            w_bounds = [(0, None) for c in range(clusters*cells)]
            w_res = minimize(w_objective, w_init.flatten(),
                    method='L-BFGS-B', jac=True, bounds=w_bounds,
                    options={'disp':disp, 'maxiter':inner_max_iters})
            w_new = w_res.x.reshape((clusters, cells))
            w_init = w_new
        w_ll = objective_fn(X, means, w_new)
        if disp:
            print('Finished updating W. Objective value: {0}'.format(w_ll))
        # step 2: given W, update M
        if method=='NoLips':
            for j in range(nolips_iters):
                m_new = update_fn(X.T, w_new.T, means.T, Xsum_m)
                m_diff = np.sqrt(np.sum((m_new.T - means)**2)/(clusters*genes))
                means = m_new.T
                if m_diff <= tol:
                    break
        elif method=='L-BFGS-B':
            m_objective = _create_m_objective(w_new, data)
            m_init = means.flatten()
            m_bounds = [(0,None) for c in range(genes*clusters)]
            m_res = minimize(m_objective, m_init,
                    method='L-BFGS-B', jac=True, bounds=m_bounds,
                    options={'disp':disp, 'maxiter':inner_max_iters})
            means = m_res.x.reshape((genes, clusters))
        m_ll = objective_fn(X, means, w_new)
        if disp:
            print('Finished updating M. Objective value: {0}'.format(m_ll))
        #ll = m_res.fun
        #m_diff = np.sqrt(np.sum((m_res.x-m_init)**2))/m_init.size
    if normalize:
        w_new = w_new/w_new.sum(0)
    return means, w_new, m_ll
