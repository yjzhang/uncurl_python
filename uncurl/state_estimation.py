# state estimation with poisson convex mixture model

from .clustering import kmeans_pp, poisson_cluster
from uncurl.nolips import nolips_update_w, objective, sparse_objective
from uncurl.nolips import sparse_nolips_update_w
# try to use parallel; otherwise
#from uncurl.nolips_parallel import sparse_nolips_update_w as parallel_sparse_nolips_update_w
try:
    from uncurl.nolips_parallel import sparse_nolips_update_w as parallel_sparse_nolips_update_w
except:
    print('Warning: cannot import sparse nolips')
    # if parallel can't be used, do not use parallel update function...
    pass
from .preprocessing import cell_normalize, log1p

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd

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
                init_W[a2, i] = (1-max_assign_weight)/(k-1)
    return init_W/init_W.sum(0)

def initialize_means(data, clusters, k):
    """
    Initializes the M matrix given the data and a set of cluster labels.
    Cluster centers are set to the mean of each cluster.

    Args:
        data (array): genes x cells
        clusters (array): 1d array of ints (0...k-1)
        k (int): number of clusters
    """
    init_w = np.zeros((data.shape[0], k))
    if sparse.issparse(data):
        for i in range(k):
            if data[:,clusters==i].shape[1]==0:
                point = np.random.randint(0, data.shape[1])
                init_w[:,i] = data[:,point].toarray().flatten()
            else:
                # memory usage might be a problem here?
                init_w[:,i] = np.array(data[:,clusters==i].mean(1)).flatten() + eps
    else:
        for i in range(k):
            if data[:,clusters==i].shape[1]==0:
                point = np.random.randint(0, data.shape[1])
                init_w[:,i] = data[:,point].flatten()
            else:
                init_w[:,i] = data[:,clusters==i].mean(1) + eps
    return init_w

def initialize_weights_nn(data, means, lognorm=True):
    """
    Initializes the weights with a nearest-neighbor approach using the means.
    """
    # TODO
    genes, cells = data.shape
    k = means.shape[1]
    if lognorm:
        data = log1p(cell_normalize(data))
    for i in range(cells):
        for j in range(k):
            pass

def _estimate_w(X, w_init, means, Xsum, update_fn, objective_fn, is_sparse=True, parallel=True, threads=4, method='NoLips', tol=1e-10, disp=False, inner_max_iters=100, output_name='W', regularization=0.0):
    clusters, cells = w_init.shape
    if method=='NoLips':
        nolips_iters = inner_max_iters
        m_sum = means.sum(0)
        lams = 1/(2*Xsum + eps)
        for j in range(nolips_iters):
            if is_sparse and parallel:
                w_new = update_fn(X.data, X.indices, X.indptr, X.shape[1], X.shape[0], means, w_init, lams, m_sum, n_threads=threads, regularization=regularization)
            else:
                w_new = update_fn(X, means, w_init, Xsum)
            #w_new = w_res.x.reshape((clusters, cells))
            #w_new = w_new/w_new.sum(0)
            if tol > 0:
                w_diff = np.sqrt(np.sum((w_new - w_init)**2)/(clusters*cells))
                if disp:
                    print('inner iter {0}: w_diff={1}'.format(j, w_diff))
                w_init = w_new
                if w_diff < tol:
                    break
            else:
                w_init = w_new
    elif method=='L-BFGS-B':
        w_objective = _create_w_objective(means, X)
        w_bounds = [(0, None) for c in range(clusters*cells)]
        w_res = minimize(w_objective, w_init.flatten(),
                method='L-BFGS-B', jac=True, bounds=w_bounds,
                options={'disp':disp, 'maxiter':inner_max_iters})
        w_new = w_res.x.reshape((clusters, cells))
        w_init = w_new
    return w_new

def _call_sparse_obj(X, M, W):
    return sparse_objective(X.data, X.indices, X.indptr, X.shape[1], X.shape[0],
            M, W)

def initialize_means_weights(data, clusters, init_means=None, init_weights=None, initialization='tsvd', max_assign_weight=0.75):
    """
    Generates initial means and weights for state estimation.
    """
    genes, cells = data.shape
    if init_means is None:
        if init_weights is not None:
            if len(init_weights.shape)==1:
                means = initialize_means(data, init_weights, clusters)
            else:
                means = initialize_means(data, init_weights.argmax(0),
                        clusters, max_assign_weight=max_assign_weight)
        elif initialization=='cluster':
            assignments, means = poisson_cluster(data, clusters)
            if init_weights is None:
                init_weights = initialize_from_assignments(assignments, clusters,
                        max_assign_weight=max_assign_weight)
        elif initialization=='kmpp':
            means, assignments = kmeans_pp(data, clusters)
        elif initialization=='km':
            km = KMeans(clusters)
            assignments = km.fit_predict(log1p(cell_normalize(data)).T)
            init_weights = initialize_from_assignments(assignments, clusters,
                    max_assign_weight)
            means = initialize_means(data, assignments, clusters)
        elif initialization=='tsvd':
            n_components = min(50, genes-1)
            #tsvd = TruncatedSVD(min(50, genes-1))
            km = KMeans(clusters)
            # remove dependence on sklearn tsvd b/c it has a bug that
            # prevents it from working properly on long inputs 
            # if num elements > 2**31
            #data_reduced = tsvd.fit_transform(log1p(cell_normalize(data)).T)
            U, Sigma, VT = randomized_svd(log1p(cell_normalize(data)).T,
                    n_components)
            data_reduced = U*Sigma
            assignments = km.fit_predict(data_reduced)
            init_weights = initialize_from_assignments(assignments, clusters,
                    max_assign_weight)
            means = initialize_means(data, assignments, clusters)
        elif initialization == 'random' or initialization == 'rand':
            # choose k random cells and set means to those
            selected_cells = np.random.choice(range(cells), size=clusters,
                    replace=False)
            means = data[:, selected_cells]
            if sparse.issparse(means):
                means = means.toarray()
    else:
        means = init_means.copy()
    means = means.astype(float)
    if init_weights is None:
        if init_means is not None:
            if initialization == 'cluster':
                assignments, means = poisson_cluster(data, clusters,
                        init=init_means, max_iters=1)
                w_init = initialize_from_assignments(assignments, clusters,
                        max_assign_weight)
            elif initialization == 'km':
                km = KMeans(clusters, init=log1p(init_means.T), max_iter=1)
                assignments = km.fit_predict(log1p(cell_normalize(data)).T)
                w_init = initialize_from_assignments(assignments, clusters,
                        max_assign_weight)
            else:
                w_init = np.random.random((clusters, cells))
                w_init = w_init/w_init.sum(0)
        else:
            w_init = np.random.random((clusters, cells))
            w_init = w_init/w_init.sum(0)
    else:
        if len(init_weights.shape)==1:
            init_weights = initialize_from_assignments(init_weights, clusters,
                    max_assign_weight)
        w_init = init_weights.copy()
    return means, w_init

def poisson_estimate_state(data, clusters, init_means=None, init_weights=None, method='NoLips', max_iters=30, tol=0, disp=False, inner_max_iters=100, normalize=True, initialization='tsvd', parallel=True, threads=4, max_assign_weight=0.75, run_w_first=True, constrain_w=False, regularization=0.0, write_progress_file=None, **kwargs):
    """
    Uses a Poisson Covex Mixture model to estimate cell states and
    cell state mixing weights.

    To lower computational costs, use a sparse matrix, set disp to False, and set tol to 0.

    Args:
        data (array): genes x cells array or sparse matrix.
        clusters (int): number of mixture components
        init_means (array, optional): initial centers - genes x clusters. Default: from Poisson kmeans
        init_weights (array, optional): initial weights - clusters x cells, or assignments as produced by clustering. Default: from Poisson kmeans
        method (str, optional): optimization method. Current options are 'NoLips' and 'L-BFGS-B'. Default: 'NoLips'.
        max_iters (int, optional): maximum number of iterations. Default: 30
        tol (float, optional): if both M and W change by less than tol (RMSE), then the iteration is stopped. Default: 1e-10
        disp (bool, optional): whether or not to display optimization progress. Default: False
        inner_max_iters (int, optional): Number of iterations to run in the optimization subroutine for M and W. Default: 100
        normalize (bool, optional): True if the resulting W should sum to 1 for each cell. Default: True.
        initialization (str, optional): If initial means and weights are not provided, this describes how they are initialized. Options: 'cluster' (poisson cluster for means and weights), 'kmpp' (kmeans++ for means, random weights), 'km' (regular k-means), 'tsvd' (tsvd(50) + k-means). Default: tsvd.
        parallel (bool, optional): Whether to use parallel updates (sparse NoLips only). Default: True
        threads (int, optional): How many threads to use in the parallel computation. Default: 4
        max_assign_weight (float, optional): If using a clustering-based initialization, how much weight to assign to the max weight cluster. Default: 0.75
        run_w_first (bool, optional): Whether or not to optimize W first (if false, M will be optimized first). Default: True
        constrain_w (bool, optional): If True, then W is normalized after every iteration. Default: False
        regularization (float, optional): Regularization coefficient for M and W. Default: 0 (no regularization).
        write_progress_file (str, optional): filename to write progress updates to.

    Returns:
        M (array): genes x clusters - state means
        W (array): clusters x cells - state mixing components for each cell
        ll (float): final log-likelihood
    """
    genes, cells = data.shape
    means, w_init = initialize_means_weights(data, clusters, init_means, init_weights, initialization, max_assign_weight)
    X = data.astype(float)
    XT = X.T
    is_sparse = False
    if sparse.issparse(X):
        is_sparse = True
        update_fn = sparse_nolips_update_w
        # convert to csc
        X = sparse.csc_matrix(X)
        XT = sparse.csc_matrix(XT)
        if parallel:
            update_fn = parallel_sparse_nolips_update_w
        Xsum = np.asarray(X.sum(0)).flatten()
        Xsum_m = np.asarray(X.sum(1)).flatten()
        # L-BFGS-B won't work right now for sparse matrices
        method = 'NoLips'
        objective_fn = _call_sparse_obj
    else:
        objective_fn = objective
        update_fn = nolips_update_w
        Xsum = X.sum(0)
        Xsum_m = X.sum(1)
        # If method is NoLips, converting to a sparse matrix
        # will always improve the performance (?) and never lower accuracy...
        if method == 'NoLips':
            is_sparse = True
            X = sparse.csc_matrix(X)
            XT = sparse.csc_matrix(XT)
            update_fn = sparse_nolips_update_w
            if parallel:
                update_fn = parallel_sparse_nolips_update_w
            objective_fn = _call_sparse_obj
    w_new = w_init
    for i in range(max_iters):
        if disp:
            print('iter: {0}'.format(i))
        if run_w_first:
            # step 1: given M, estimate W
            w_new = _estimate_w(X, w_new, means, Xsum, update_fn, objective_fn, is_sparse, parallel, threads, method, tol, disp, inner_max_iters, 'W', regularization)
            if constrain_w:
                w_new = w_new/w_new.sum(0)
            if disp:
                w_ll = objective_fn(X, means, w_new)
                print('Finished updating W. Objective value: {0}'.format(w_ll))
            # step 2: given W, update M
            means = _estimate_w(XT, means.T, w_new.T, Xsum_m, update_fn, objective_fn, is_sparse, parallel, threads, method, tol, disp, inner_max_iters, 'M', regularization)
            means = means.T
            if disp:
                w_ll = objective_fn(X, means, w_new)
                print('Finished updating M. Objective value: {0}'.format(w_ll))
        else:
            # step 1: given W, update M
            means = _estimate_w(XT, means.T, w_new.T, Xsum_m, update_fn, objective_fn, is_sparse, parallel, threads, method, tol, disp, inner_max_iters, 'M', regularization)
            means = means.T
            if disp:
                w_ll = objective_fn(X, means, w_new)
                print('Finished updating M. Objective value: {0}'.format(w_ll))
            # step 2: given M, estimate W
            w_new = _estimate_w(X, w_new, means, Xsum, update_fn, objective_fn, is_sparse, parallel, threads, method, tol, disp, inner_max_iters, 'W', regularization)
            if constrain_w:
                w_new = w_new/w_new.sum(0)
            if disp:
                w_ll = objective_fn(X, means, w_new)
                print('Finished updating W. Objective value: {0}'.format(w_ll))
        # write progress to progress file
        if write_progress_file is not None:
            progress = open(write_progress_file, 'w')
            progress.write(str(i))
            progress.close()
    if normalize:
        w_new = w_new/w_new.sum(0)
    m_ll = objective_fn(X, means, w_new)
    return means, w_new, m_ll


def update_m(data, old_M, old_W, selected_genes, disp=False, inner_max_iters=100, parallel=True, threads=4, write_progress_file=None, tol=0.0, regularization=0.0, **kwargs):
    """
    This returns a new M matrix that contains all genes, given an M that was
    created from running state estimation with a subset of genes.

    Args:
        data (sparse matrix or dense array): data matrix of shape (genes, cells), containing all genes
        old_M (array): shape is (selected_genes, k)
        old_W (array): shape is (k, cells)
        selected_genes (list): list of selected gene indices
        Rest of the args are as in poisson_estimate_state

    Returns:
        new_M: array of shape (all_genes, k)
    """
    genes, cells = data.shape
    k = old_M.shape[1]
    non_selected_genes = [x for x in range(genes) if x not in set(selected_genes)]
    # 1. initialize new M
    new_M = np.zeros((genes, k))
    new_M[selected_genes, :] = old_M
    # TODO: how to initialize rest of genes?
    # data*w?
    if disp:
        print('computing initial guess for M by data*W.T')
    if not sparse.issparse(data):
        data = sparse.csc_matrix(data)
    new_M_non_selected = data[non_selected_genes, :] * sparse.csc_matrix(old_W.T)
    new_M[non_selected_genes, :] = new_M_non_selected.toarray()
    X = data.astype(float)
    XT = X.T
    is_sparse = False
    if sparse.issparse(X):
        is_sparse = True
        update_fn = sparse_nolips_update_w
        # convert to csc
        X = sparse.csc_matrix(X)
        XT = sparse.csc_matrix(XT)
        if parallel:
            update_fn = parallel_sparse_nolips_update_w
        Xsum_m = np.asarray(X.sum(1)).flatten()
        # L-BFGS-B won't work right now for sparse matrices
        method = 'NoLips'
        objective_fn = _call_sparse_obj
    else:
        objective_fn = objective
        update_fn = nolips_update_w
        Xsum_m = X.sum(1)
        # If method is NoLips, converting to a sparse matrix
        # will always improve the performance (?) and never lower accuracy...
        # will almost always improve performance?
        # if sparsity is below 40%?
        if method == 'NoLips':
            is_sparse = True
            X = sparse.csc_matrix(X)
            XT = sparse.csc_matrix(XT)
            update_fn = sparse_nolips_update_w
            if parallel:
                update_fn = parallel_sparse_nolips_update_w
            objective_fn = _call_sparse_obj
    if disp:
        print('starting estimating M')
    new_M = _estimate_w(XT, new_M.T, old_W.T, Xsum_m, update_fn, objective_fn, is_sparse, parallel, threads, method, tol, disp, inner_max_iters, 'M', regularization)
    if write_progress_file is not None:
        progress = open(write_progress_file, 'w')
        progress.write('0')
        progress.close()
    return new_M.T
