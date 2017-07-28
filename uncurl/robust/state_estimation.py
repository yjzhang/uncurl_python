# state estimation with poisson convex mixture model

from uncurl.clustering import kmeans_pp

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

eps=1e-8


def _create_poiss_w_objective(m, X):
    """
    Creates an objective function and its derivative for W, given M and X (data)

    Args:
        m (array): genes x clusters
        X (array): genes x cells
        selected_genes (array): array of ints - genes to be selected
    """
    # TODO: excluded genes
    genes, clusters = m.shape
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

def _create_poiss_m_objective(w, X):
    """
    Creates an objective function and its derivative for M, given W and X

    Args:
        w (array): clusters x cells
        X (array): genes x cells
        selected_genes (array): array of ints - genes to be selected
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

def _poisson_calculate_lls(X, M, W):
    """
    For hard thresholding: this calculates the log-likelihood of each
    gene, and returns a list of log-likelihoods.
    """
    genes, cells = X.shape
    L = np.zeros(genes)
    d = M.dot(W)
    l2 = gammaln(X+1)
    print l2
    for i in range(genes):
        L[i] = np.sum(X[i,:]*np.log(d[i,:]) - d[i,:] - l2[i,:])
        if np.isnan(L[i]):
            L[i] = -np.inf
    return L

def initialize_from_assignments(assignments, k):
    """
    Creates a weight initialization matrix from Poisson clustering assignments.

    Args:
        assignments (array): 1D array of integers, of length cells
        k (int): number of states/clusters

    Returns:
        init_W (array): k x cells
    """
    cells = len(assignments)
    init_W = np.zeros((k, cells))
    for i, a in enumerate(assignments):
        # entirely arbitrary... maybe it would be better to scale
        # the weights based on k?
        init_W[a, i] = 0.75
        for a2 in range(k):
            if a2!=a:
                init_W[a2, i] = 0.25/(k-1)
    return init_W

def one_round(data, M, W, selected_genes):
    pass

# TODO: add hard thresholding
def robust_estimate_state(data, clusters, dist='Poiss', init_means=None, init_weights=None, max_iters=10, tol=1e-4, disp=True, inner_max_iters=400, reps=1, normalize=True, gene_portion=0.2):
    """
    Uses a Poisson Covex Mixture model to estimate cell states and
    cell state mixing weights.

    Args:
        data (array): genes x cells
        clusters (int): number of mixture components
        dist (string, optional): Distribution used - only 'Poiss' is implemented. Default: 'Poiss'
        init_means (array, optional): initial centers - genes x clusters. Default: kmeans++ initializations
        init_weights (array, optional): initial weights - clusters x cells, or assignments as produced by clustering. Default: random(0,1)
        max_iters (int, optional): maximum number of iterations. Default: 10
        tol (float, optional): if both M and W change by less than tol, then the iteration is stopped. Default: 1e-4
        disp (bool, optional): whether or not to display optimization parameters. Default: True
        inner_max_iters (int, optional): Number of iterations to run in the scipy minimizer for M and W. Default: 400
        reps (int, optional): number of random initializations. Default: 1.
        normalize (bool, optional): True if the resulting W should sum to 1 for each cell. Default: True.
        gene_portion (float, optional): The proportion of genes to use for estimating W after hard thresholding. Default: 0.2

    Returns:
        M (array): genes x clusters - state means
        W (array): clusters x cells - state mixing components for each cell
        ll (float): final log-likelihood
        genes (array): 1d array of all genes used in final iteration.
    """
    genes, cells = data.shape
    if init_means is None:
        means, assignments = kmeans_pp(data, clusters)
    else:
        means = init_means.copy()
    clusters = means.shape[1]
    w_init = np.random.random(cells*clusters)
    if init_weights is not None:
        if len(init_weights.shape)==1:
            init_weights = initialize_from_assignments(init_weights, clusters)
        w_init = init_weights.reshape(cells*clusters)
    m_init = means.reshape(genes*clusters)
    # repeat steps 1 and 2 until convergence:
    ll = np.inf
    # objective functions...
    w_obj = _create_poiss_w_objective
    m_obj = _create_poiss_m_objective
    ll_func = _poisson_calculate_lls
    included_genes = np.arange(genes)
    num_genes = int(np.ceil(gene_portion*genes))
    for i in range(max_iters):
        if disp:
            print('iter: {0}'.format(i))
        w_bounds = [(0, None) for x in w_init]
        m_bounds = [(0, None) for x in m_init]
        # step 1: given M, estimate W
        w_objective = w_obj(means[included_genes,:], data[included_genes,:])
        # TODO: select between L-BFGS-B or SLSQP optimization methods
        w_res = minimize(w_objective, w_init, method='L-BFGS-B', jac=True, bounds=w_bounds, options={'disp':disp, 'maxiter':inner_max_iters})
        w_diff = np.sqrt(np.sum((w_res.x-w_init)**2))/w_init.size
        w_new = w_res.x.reshape((clusters, cells))
        w_init = w_res.x
        # step 2: given W, update M
        m_objective = m_obj(w_new, data)
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
        # step 3: hard thresholding/gene subset selection
        lls = ll_func(data, m_new, w_new)
        included_genes = lls.argsort()[::-1][:num_genes]
        if disp:
            print(lls[included_genes])
            print(sum(~np.isnan(lls)))
            print(included_genes)
    if normalize:
        w_new = w_new/w_new.sum(0)
    return m_new, w_new, ll, included_genes
