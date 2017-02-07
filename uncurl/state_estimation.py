# state estimation with poisson convex mixture model

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
        #print 'eval_w_obj'
        # convert w into a matrix first... because it's a vector for
        # optimization purposes
        w = w.reshape((m.shape[1], X.shape[1]))
        d = m.dot(w)+eps
        return np.sum(d - X*np.log(d))
    def deriv(w):
        # TODO
        #print 'eval_w_deriv'
        # derivative of objective wrt all elements of w
        # for w_{ij}, the derivative is... m_j1+...+m_jn sum over genes minus 
        # x_ij
        w2 = w.reshape((m.shape[1], X.shape[1]))
        d = m.dot(w2)+eps
        deriv = np.zeros(w.shape)
        for i in range(clusters):
            for j in range(cells):
                deriv[i*cells+j] = sum(m[:,i]) - sum(m[:,i]*X[:,j]/d[:,j])
        return deriv
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
        #print 'eval_m_obj'
        m = m.reshape((X.shape[0], w.shape[0]))
        d = m.dot(w)+eps
        return np.sum(d - X*np.log(d))
    def deriv(m):
        #print 'eval_m_deriv'
        m2 = m.reshape((X.shape[0], w.shape[0]))
        d = m2.dot(w)+eps
        deriv = np.zeros(m.shape)
        for i in range(genes):
            for j in range(clusters):
                deriv[i*clusters+j] = sum(w[j,:]) - sum(w[j,:]*X[i,:]/d[i,:])
        return deriv
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

def poisson_estimate_state(data, means, init_weights=None, max_iters=10, tol=1e-6):
    """
    Uses a Poisson Covex Mixture model to estimate cell states and
    cell state mixing.

    Args:
        data (array): genes x cells
        means (array): initial centers - genes x clusters
        init_weights (array): initial weights - clusters x cells
        max_iters (int): maximum number of iterations
        tol (float): if both M and W change by less than tol, then the iteration
            is stopped.

    Returns:
        two matrices, M and W: M is genes x clusters, W is clusters x cells.
        M is the state centers and W is the state mixing components for each
        cell.
    """
    genes, cells = data.shape
    clusters = means.shape[1]
    w_constraints = _create_w_constraints(cells, clusters)
    w_init = np.random.random(cells*clusters)
    if init_weights is not None:
        w_init = init_weights.reshape(cells*clusters)
    m_init = means.reshape(genes*clusters)
    # repeat steps 1 and 2 until convergence:
    for i in range(max_iters):
        print('iter: {0}'.format(i))
        w_bounds = [(0, 1.0) for x in w_init]
        m_bounds = [(0, None) for x in m_init]
        # step 1: given M, estimate W
        w_objective, w_deriv = _create_w_objective(means, data)
        w_res = minimize(w_objective, w_init, method='SLSQP', jac=w_deriv, bounds=w_bounds, constraints=w_constraints, options={'disp':True, 'maxiter':100})
        w_diff = np.sqrt(np.sum((w_res.x-w_init)**2))
        w_new = w_res.x.reshape((clusters, cells))
        w_init = w_res.x
        # step 2: given W, update M
        m_objective, m_deriv = _create_m_objective(w_new, data)
        # method could be 'L-BFGS-B' or 'SLSQP'... SLSQP gives a memory error...
        m_res = minimize(m_objective, m_init, method='TNC', jac=m_deriv, bounds=m_bounds, options={'disp':True, 'maxiter':1000})
        m_diff = np.sqrt(np.sum((m_res.x-m_init)**2))
        m_new = m_res.x.reshape((genes, clusters))
        m_init = m_res.x
        means = m_new
        if w_diff < tol and m_diff < tol:
            break
    return m_new, w_new
