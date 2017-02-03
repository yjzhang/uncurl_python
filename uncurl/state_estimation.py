# state estimation with poisson convex mixture model

import numpy as np
from scipy.optimize import minimize

def _create_w_objective(m, X):
    """
    Creates an objective function and its derivative for W, given M and X (data)

    Args:
        m (array): genes x clusters
        X (array): genes x cells
    """
    def objective(w):
        # convert w into a matrix first... because it's a vector for
        # optimization purposes
        w = w.reshape((m.shape[1], X.shape[1]))
        d = m.dot(w)
        return np.sum(d - X*d)
    def deriv(w):
        # TODO
        # derivative of objective wrt all elements of w
        # for w_{ij}, the derivative is... m_j1+...+m_jn sum over genes minus 
        # x_ij
        pass
    return objective

def _create_m_objective(w, X):
    """
    Creates an objective function and its derivative for M, given W and X

    Args:
        w (array): clusters x cells
        X (array): genes x cells
    """
    def objective(m):
        m = m.reshape((X.shape[0], w.shape[1]))
        d = m.dot(w)
        return np.sum(d - X*d)
    def deriv(m):
        # TODO
        pass
    return objective

def _w_equality_constraints(cells, clusters, i):
    fun = lambda w: sum(w[i*clusters:(i+1)*(clusters)]) - 1
    jac_matrix = np.zeros(cells*clusters)
    jac_matrix[i*clusters : (i+1)*clusters] = 1.0
    jac = lambda w: jac_matrix
    return (fun, jac)

def _create_w_constraints(cells, clusters):
    def equality_constraint():
        for i in range(cells):
            yield _w_equality_constraints(cells, clusters, i)
    eq_constraints = [{'type': 'eq', 'fun': x, 'jac': y} for x,y in equality_constraint()]
    return tuple(eq_constraints)

def poisson_estimate_state(data, means, max_iters=10):
    """
    Uses a Poisson Covex Mixture model to estimate cell states and
    cell state mixing.

    Args:
        data (array): genes x cells
        means (array): initial centers - genes x clusters
        max_iters (int): maximum number of iterations

    Returns:
        two matrices, M and W: M is genes x clusters, W is clusters x cells.
        M is the cluster states and W is the state mixing components for each
        cell.
    """
    genes, cells = data.shape
    clusters = means.shape[1]
    w_constraints = _create_w_constraints(cells, clusters)
    w_init = np.random.randn(cells*clusters)
    m_init = means.reshape(genes*clusters)
    w_bounds = [(0, None) for x in w_init]
    m_bounds = [(0, None) for x in m_init]
    # repeat steps 1 and 2 until convergence:
    for i in range(max_iters):
        # step 1: given M, estimate W
        w_objective = _create_w_objective(means, data)
        w_res = minimize(w_objective, w_init, bounds=w_bounds, constraints=w_constraints)
        w_new = w_res.x.reshape((clusters, cells))
        w_init = w_res.x
        # step 2: given W, update M
        m_objective = _create_m_objective(w_new, data)
        m_res = minimize(m_objective, m_init, bounds=m_bounds)
        m_new = m_res.x.reshape((genes, clusters))
        m_init = m_res.x
        means = m_new
    return m_new, w_new
