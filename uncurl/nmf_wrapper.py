# wrapper for various NMF methods

from scipy import sparse
from sklearn.decomposition import NMF

from .preprocessing import log1p, cell_normalize

# TODO: initialization
def log_norm_nmf(data, k, normalize_h=True, return_cost=True, **kwargs):
    """
    Args:
        data (array): dense or sparse array with shape (genes, cells)
        k (int): number of cell types
        normalize_h (bool): True if H should be normalized (so that each column sums to 1)
        return_cost (bool): True if the NMF objective value (squared error) should be returned. Default: True
        **kwargs: arguments to NMF

    Returns:
        Two matrices W of shape (genes, k) and H of shape (k, cells). If return_cost is True (which it is by default), then the cost will also be returned.
    """
    nmf = NMF(k, **kwargs)
    data = log1p(cell_normalize(data))
    W = nmf.fit_transform(data)
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
