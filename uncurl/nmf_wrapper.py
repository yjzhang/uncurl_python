# wrapper for various NMF methods

import numpy as np
from scipy import sparse
from sklearn.decomposition import NMF, non_negative_factorization

from .preprocessing import log1p, cell_normalize
from .state_estimation import initialize_from_assignments

def nmf_init(data, clusters, k, init='enhanced'):
    """
    Generates initial W and H given a data set and an array of cluster labels.

    There are 3 options for init:
        enhanced - uses EIn-NMF from Gong 2013
        basic - uses means for W, assigns H such that the chosen cluster for a given cell has value 0.75 and all others have 0.25/(k-1).
        nmf - uses means for W, and assigns H using the NMF objective while holding W constant.
    """
    init_w = np.zeros((data.shape[0], k))
    if sparse.issparse(data):
        for i in range(k):
            if data[:,clusters==i].shape[1]==0:
                point = np.random.randint(0, data.shape[1])
                init_w[:,i] = data[:,point].toarray().flatten()
            else:
                init_w[:,i] = np.array(data[:,clusters==i].mean(1)).flatten()
    else:
        for i in range(k):
            if data[:,clusters==i].shape[1]==0:
                point = np.random.randint(0, data.shape[1])
                init_w[:,i] = data[:,point].flatten()
            else:
                init_w[:,i] = data[:,clusters==i].mean(1)
    init_h = np.zeros((k, data.shape[1]))
    if init == 'enhanced':
        distances = np.zeros((k, data.shape[1]))
        for i in range(k):
            for j in range(data.shape[1]):
                distances[i,j] = np.sqrt(((data[:,j] - init_w[:,i])**2).sum())
        for i in range(k):
            for j in range(data.shape[1]):
                init_h[i,j] = 1/((distances[:,j]/distances[i,j])**(-2)).sum()
    elif init == 'basic':
        init_h = initialize_from_assignments(clusters, k)
    elif init == 'nmf':
        init_h_, _, n_iter = non_negative_factorization(data.T, n_components=k, init='custom', update_H=False, H=init_w.T)
        init_h = init_h_.T
    return init_w, init_h

# TODO: initialization
def log_norm_nmf(data, k, normalize_h=True, return_cost=True, init_h=None, init_w=None, **kwargs):
    """
    Args:
        data (array): dense or sparse array with shape (genes, cells)
        k (int): number of cell types
        normalize_h (bool, optional): True if H should be normalized (so that each column sums to 1). Default: True
        return_cost (bool, optional): True if the NMF objective value (squared error) should be returned. Default: True
        init_h (array, optional): Initial value for H. Default: None
        init_w (array, optional): Initial value for W. Default: None
        **kwargs: misc arguments to NMF

    Returns:
        Two matrices W of shape (genes, k) and H of shape (k, cells). They correspond to M and W in Poisson state estimation. If return_cost is True (which it is by default), then the cost will also be returned. This might be prohibitably costly
    """
    init = None
    if init_h is not None or init_w is not None:
        init = 'custom'
        if init_h is None:
            init_h_, _, n_iter = non_negative_factorization(data.T, n_components=k, init='custom', update_H=False, H=init_w.T)
            init_h = init_h_.T
        elif init_w is None:
            init_w, _, n_iter = non_negative_factorization(data, n_components=k, init='custom', update_H=False, H=init_h)
    nmf = NMF(k, init=init, **kwargs)
    data = log1p(cell_normalize(data))
    W = nmf.fit_transform(data, W=init_w, H=init_h)
    H = nmf.components_
    if normalize_h:
        H = H/H.sum(0)
    if return_cost:
        cost = 0
        if sparse.issparse(data):
            ws = sparse.csr_matrix(W)
            hs = sparse.csr_matrix(H)
            cost = 0.5*((data - ws.dot(hs)).power(2)).sum()
        else:
            cost = 0.5*((data - W.dot(H))**2).sum()
        return W, H, cost
    else:
        return W, H

def norm_nmf(data, k, normalize_h=True, **kwargs):
    """
    Args:
        data (array): dense or sparse array with shape (genes, cells)
        k (int): number of cell types
        normalize_h (bool): True if H should be normalized (so that each column sums to 1)
        **kwargs: arguments to NMF

    Returns:
        Two matrices W of shape (genes, k) and H of shape (k, cells)
    """
    nmf = NMF(k, **kwargs)
    data = cell_normalize(data)
    W = nmf.fit_transform(data)
    H = nmf.components_
    if normalize_h:
        H = H/H.sum(0)
    return W, H
