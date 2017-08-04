# state estimation with poisson convex mixture model

from uncurl.clustering import kmeans_pp
from uncurl.state_estimation import initialize_from_assignments, nolips_update_w

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

eps=1e-10


def _create_poiss_w_objective(m, X):
    """
    Creates an objective function and its derivative for W, given M and X (data)

    Args:
        m (array): genes x clusters
        X (array): genes x cells
    """
    # TODO: excluded genes
    genes, clusters = m.shape
    m_sum = m.sum(0)
    ms2 = m_sum.reshape((clusters, 1))
    const = gammaln(X+1)
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
        deriv = ms2 - m2
        return np.sum(d - X*np.log(d) + const)/genes, deriv.flatten()/genes
    return objective

def _create_poiss_m_objective(w, X):
    """
    Creates an objective function and its derivative for M, given W and X

    Args:
        w (array): clusters x cells
        X (array): genes x cells
        included_genes (array): list of included genes
    """
    clusters, cells = w.shape
    genes = X.shape[0]
    w_sum = w.sum(1)
    const = gammaln(X+1)
    def objective(m):
        m = m.reshape((genes, clusters))
        d = m.dot(w)+eps
        s = d - X*np.log(d) + const
        temp = X/d
        w2 = w.dot(temp.T)
        deriv = w_sum - w2.T
        return np.sum(s)/genes, deriv.flatten()/genes
    return objective

def _poisson_calculate_lls(X, M, W, use_constant=True, add_eps=True):
    """
    For hard thresholding: this calculates the log-likelihood of each
    gene, and returns a list of log-likelihoods.
    """
    genes, cells = X.shape
    L = np.zeros(genes)
    d = M.dot(W)
    if add_eps:
        d += 1e-30
    # d[d==0] = np.min(d[d>0])/1e4
    LLs = X*np.log(d) - d
    if use_constant:
        LLs -= gammaln(X+1)
    L = np.sum(LLs, 1)
    L[np.isnan(L)] = -np.inf
    return L

def one_round(data, M, W, selected_genes):
    pass

def robust_estimate_state(data, clusters, dist='Poiss', init_means=None, init_weights=None, max_iters=10, tol=1e-4, disp=True, inner_max_iters=25, reps=1, normalize=True, gene_portion=0.2, use_constant=True):
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
    w_init = np.random.random((clusters, cells))
    if init_weights is not None:
        if len(init_weights.shape)==1:
            init_weights = initialize_from_assignments(init_weights, clusters)
    # repeat steps 1 and 2 until convergence:
    ll = np.inf
    # objective functions...
    w_obj = _create_poiss_w_objective
    m_obj = _create_poiss_m_objective
    ll_func = _poisson_calculate_lls
    included_genes = np.arange(genes)
    num_genes = int(np.ceil(gene_portion*genes))
    if disp:
        print('num_genes: {0}'.format(num_genes))
    nolips_iters = 25
    for i in range(max_iters):
        if disp:
            print('iter: {0}'.format(i))
        w_bounds = [(0, None) for x in w_init]
        # step 1: given M, estimate W
        w_objective = w_obj(means[included_genes,:], data[included_genes,:])
        # convert to nolips
        for i in range(nolips_iters):
            w_new = nolips_update_w(data[included_genes,:]+eps, means[included_genes,:], w_init)
            #w_new = w_res.x.reshape((clusters, cells))
            #w_new = w_new/w_new.sum(0)
            w_init = w_new
        w_ll, w_deriv = w_objective(w_new.reshape(clusters*cells))
        #w_res = minimize(w_objective, w_init, method='L-BFGS-B', jac=True, bounds=w_bounds, options={'disp':disp, 'maxiter':inner_max_iters})
        #w_diff = np.sqrt(np.sum((w_res.x-w_init)**2))/w_init.size
        #w_init = w_res.x
        #w_new = w_res.x.reshape((clusters, cells))
        # step 2: given W, update M
        w_ll, w_deriv = w_objective(w_new.reshape(clusters*cells))
        if disp:
            print('Finished updating W. Objective value: {0}'.format(w_ll))
        # step 2: given W, update M
        for i in range(nolips_iters):
            m_new = nolips_update_w(data[included_genes,:].T+eps, w_new.T, means[included_genes,:].T)
            means[included_genes,:] = m_new.T
        m_objective = m_obj(w_new, data[included_genes,:])
        m_ll, m_deriv = m_objective(means.reshape(genes*clusters))
        if disp:
            print('Finished updating M. Objective value: {0}'.format(m_ll))

        # step 3: hard thresholding/gene subset selection
        lls = ll_func(data, means, w_new, use_constant)
        if i < max_iters - 1:
            included_genes = lls.argsort()[::-1][:num_genes]
            if disp:
                print(lls[included_genes])
                print(sum(~np.isinf(lls)))
                print(included_genes)
                print('selected number of genes: ' + str(len(included_genes)))
    if normalize:
        w_new = w_new/w_new.sum(0)
    return means, w_new, ll, included_genes
