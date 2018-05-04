# wrapper for various NMF methods

import numpy as np
from scipy import sparse
from sklearn.decomposition import NMF, non_negative_factorization

from .preprocessing import log1p, cell_normalize
from .state_estimation import initialize_from_assignments

def nmf_init(data, clusters, k, init='enhanced'):
    """
    Generates initial M and W given a data set and an array of cluster labels.

    There are 3 options for init:
        enhanced - uses EIn-NMF from Gong 2013
        basic - uses means for M, assigns W such that the chosen cluster for a given cell has value 0.75 and all others have 0.25/(k-1).
        nmf - uses means for M, and assigns W using the NMF objective while holding M constant.
    """
    init_m = np.zeros((data.shape[0], k))
    if sparse.issparse(data):
        for i in range(k):
            if data[:,clusters==i].shape[1]==0:
                point = np.random.randint(0, data.shape[1])
                init_m[:,i] = data[:,point].toarray().flatten()
            else:
                init_m[:,i] = np.array(data[:,clusters==i].mean(1)).flatten()
    else:
        for i in range(k):
            if data[:,clusters==i].shape[1]==0:
                point = np.random.randint(0, data.shape[1])
                init_m[:,i] = data[:,point].flatten()
            else:
                init_m[:,i] = data[:,clusters==i].mean(1)
    init_w = np.zeros((k, data.shape[1]))
    if init == 'enhanced':
        distances = np.zeros((k, data.shape[1]))
        for i in range(k):
            for j in range(data.shape[1]):
                distances[i,j] = np.sqrt(((data[:,j] - init_m[:,i])**2).sum())
        for i in range(k):
            for j in range(data.shape[1]):
                init_w[i,j] = 1/((distances[:,j]/distances[i,j])**(-2)).sum()
    elif init == 'basic':
        init_w = initialize_from_assignments(clusters, k)
    elif init == 'nmf':
        init_w_, _, n_iter = non_negative_factorization(data.T, n_components=k, init='custom', update_W=False, W=init_m.T)
        init_w = init_w_.T
    return init_m, init_w

# TODO: initialization if init_w is a cluster list?
def log_norm_nmf(data, k, normalize_w=True, return_cost=True, init_weights=None, init_means=None, write_progress_file=None, **kwargs):
    """
    Args:
        data (array): dense or sparse array with shape (genes, cells)
        k (int): number of cell types
        normalize_w (bool, optional): True if W should be normalized (so that each column sums to 1). Default: True
        return_cost (bool, optional): True if the NMF objective value (squared error) should be returned. Default: True
        init_weights (array, optional): Initial value for W. Default: None
        init_means (array, optional): Initial value for M. Default: None
        **kwargs: misc arguments to NMF

    Returns:
        Two matrices M of shape (genes, k) and W of shape (k, cells). They correspond to M and M in Poisson state estimation. If return_cost is True (which it is by default), then the cost will also be returned. This might be prohibitably costly
    """
    init = None
    data = log1p(cell_normalize(data))
    if init_weights is not None or init_means is not None:
        init = 'custom'
        if init_weights is None:
            init_weights_, _, n_iter = non_negative_factorization(data.T, n_components=k, init='custom', update_W=False, W=init_means.T)
            init_weights = init_weights_.T
        elif init_means is None:
            init_means, _, n_iter = non_negative_factorization(data, n_components=k, init='custom', update_W=False, W=init_weights)
        init_means = init_means.copy(order='C')
        init_weights = init_weights.copy(order='C')
    nmf = NMF(k, init=init, **kwargs)
    if write_progress_file is not None:
        progress = open(write_progress_file, 'w')
        progress.write(str(0))
        progress.close()
    M = nmf.fit_transform(data, W=init_means, H=init_weights)
    W = nmf.components_
    if normalize_w:
        W = W/W.sum(0)
    if return_cost:
        cost = 0
        if sparse.issparse(data):
            ws = sparse.csr_matrix(M)
            hs = sparse.csr_matrix(W)
            cost = 0.5*((data - ws.dot(hs)).power(2)).sum()
        else:
            cost = 0.5*((data - M.dot(W))**2).sum()
        return M, W, cost
    else:
        return M, W

# TODO: initialization
def norm_nmf(data, k, init_weights=None, init_means=None, normalize_w=True, return_cost=True, write_progress_file=None, **kwargs):
    """
    Args:
        data (array): dense or sparse array with shape (genes, cells)
        k (int): number of cell types
        normalize_w (bool): True if W should be normalized (so that each column sums to 1)
        init_weights (array, optional): Initial value for W. Default: None
        init_means (array, optional): Initial value for M. Default: None
        **kwargs: misc arguments to NMF

    Returns:
        Two matrices M of shape (genes, k) and W of shape (k, cells)
    """
    data = cell_normalize(data)
    init = None
    if init_weights is not None or init_means is not None:
        init = 'custom'
        if init_weights is None:
            init_weights_, _, n_iter = non_negative_factorization(data.T, n_components=k, init='custom', update_W=False, W=init_means.T)
            init_weights = init_weights_.T
        elif init_means is None:
            init_means, _, n_iter = non_negative_factorization(data, n_components=k, init='custom', update_W=False, W=init_weights)
        init_means = init_means.copy(order='C')
        init_weights = init_weights.copy(order='C')
    nmf = NMF(k, init=init, **kwargs)
    if write_progress_file is not None:
        progress = open(write_progress_file, 'w')
        progress.write(str(0))
        progress.close()
    M = nmf.fit_transform(data, W=init_means, H=init_weights)
    W = nmf.components_
    if normalize_w:
        W = W/W.sum(0)
    if return_cost:
        cost = 0
        if sparse.issparse(data):
            ws = sparse.csr_matrix(M)
            hs = sparse.csr_matrix(W)
            cost = 0.5*((data - ws.dot(hs)).power(2)).sum()
        else:
            cost = 0.5*((data - M.dot(W))**2).sum()
        return M, W, cost
    else:
        return M, W


