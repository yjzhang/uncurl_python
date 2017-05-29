# state estimation with poisson convex mixture model
# TODO: not yet implemented

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
        return np.sum(d - X*np.log(d))/genes
    def deriv(w):
        # derivative of objective wrt all elements of w
        # for w_{ij}, the derivative is... m_j1+...+m_jn sum over genes minus 
        # x_ij
        w2 = w.reshape((m.shape[1], X.shape[1]))
        d = m.dot(w2)+eps
        temp = X/d
        m_sum = m.sum(0)
        m2 = m.T.dot(temp)
        deriv = m_sum.reshape((clusters, 1)) - m2
        return deriv.flatten()/genes
    return objective, deriv

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
        return np.sum(d - X*np.log(d))/genes
    def deriv(m):
        m2 = m.reshape((X.shape[0], w.shape[0]))
        d = m2.dot(w)+eps
        temp = X/d
        w_sum = w.sum(1)
        w2 = w.dot(temp.T)
        deriv = w_sum - w2.T
        return deriv.flatten()/genes
    return objective, deriv

def _w_equality_constraints(cells, clusters, i):
    # constraint for the ith cell
    fun = lambda w: w.reshape((clusters, cells)).sum(0)[i] - 1.0
    jac_matrix = np.zeros(cells*clusters)
    for j in range(clusters):
        jac_matrix[j*cells + i] = 1.0
    jac = lambda w: jac_matrix
    return (fun, jac)

def _create_w_constraints(cells, clusters):
    def equality_constraint():
        for i in range(cells):
            yield _w_equality_constraints(cells, clusters, i)
    eq_constraints = [{'type': 'eq', 'fun': x, 'jac': y} for x,y in equality_constraint()]
    return tuple(eq_constraints)

def nb_estimate_state(data, clusters, R, init_means=None, init_weights=None, max_iters=10, tol=1e-4, disp=True, inner_max_iters=400):
    """
    Uses a Negative Binomial Mixture model to estimate cell states and
    cell state mixing weights.

    Args:
        data (array): genes x cells
        clusters (int): number of mixture components
        R (array): vector of length genes containing the dispersion estimates for each gene.
        init_means (array, optional): initial centers - genes x clusters. Default: kmeans++ initializations
        init_weights (array, optional): initial weights - clusters x cells. Default: random(0,1)
        max_iters (int, optional): maximum number of iterations. Default: 10
        tol (float, optional): if both M and W change by less than tol, then the iteration is stopped. Default: 1e-4
        disp (bool, optional): whether or not to display optimization parameters. Default: True
        inner_max_iters (int, optional): Number of iterations to run in the scipy minimizer for M and W. Default: 400

    Returns:
        two matrices, M and W: M is genes x clusters, W is clusters x cells.
        M is the state centers and W is the state mixing components for each
        cell.
    """
    genes, cells = data.shape
    if init_means is None:
        means, assignments = kmeans_pp(data, clusters)
    else:
        means = init_means.copy()
    clusters = means.shape[1]
    #w_constraints = _create_w_constraints(cells, clusters)
    w_init = np.random.random(cells*clusters)
    if init_weights is not None:
        w_init = init_weights.reshape(cells*clusters)
    m_init = means.reshape(genes*clusters)
    # repeat steps 1 and 2 until convergence:
    for i in range(max_iters):
        if disp:
            print('iter: {0}'.format(i))
        w_bounds = [(0, 1.0) for x in w_init]
        m_bounds = [(0, None) for x in m_init]
        # step 1: given M, estimate W
        w_objective, w_deriv = _create_w_objective(means, data)
        w_res = minimize(w_objective, w_init, method='SLSQP', jac=w_deriv, bounds=w_bounds, options={'disp':disp, 'maxiter':inner_max_iters})
        w_diff = np.sqrt(np.sum((w_res.x-w_init)**2))/w_init.size
        w_new = w_res.x.reshape((clusters, cells))
        w_init = w_res.x
        # step 2: given W, update M
        m_objective, m_deriv = _create_m_objective(w_new, data)
        # method could be 'L-BFGS-B' or 'SLSQP'... SLSQP gives a memory error...
        # or use TNC...
        m_res = minimize(m_objective, m_init, method='L-BFGS-B', jac=m_deriv, bounds=m_bounds, options={'disp':disp, 'maxiter':inner_max_iters})
        m_diff = np.sqrt(np.sum((m_res.x-m_init)**2))/m_init.size
        m_new = m_res.x.reshape((genes, clusters))
        m_init = m_res.x
        means = m_new
        if w_diff < tol and m_diff < tol:
            break
    w_new = w_new/w_new.sum(0)
    return m_new, w_new
