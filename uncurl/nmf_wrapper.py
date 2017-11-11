# wrapper for various NMF methods

from sklearn.decomposition import NMF

from .preprocessing import log1p, cell_normalize

def log_norm_nmf(data, k, normalize_h=True, **kwargs):
    """
    Args:
        data (array): dense or sparse array with shape (genes, cells)
        k (int): number of cell types
        normalize_h (bool): True if H should be normalized (so that each column sums to 1)
        **kwargs: arguments to NMF

    Returns:
        Two matrices W of shape (genes, k) and H of shape (k, cells)
    """
    nmf = NMF(k)
    data = log1p(cell_normalize(data))
    W = nmf.fit_transform(data)
    H = nmf.components_
    if normalize_h:
        H = H/H.sum(0)
    return W, H
