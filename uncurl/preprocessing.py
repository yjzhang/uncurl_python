"""
Misc functions...
"""

import numpy as np
from scipy import sparse

from uncurl.sparse_utils import sparse_cell_normalize, sparse_means_var_csc

def sparse_mean_var(data):
    """
    Calculates the variance for each row of a sparse matrix,
    using the relationship Var = E[x^2] - E[x]^2.

    Returns:
        pair of matrices mean, variance.
    """
    data = sparse.csc_matrix(data)
    return sparse_means_var_csc(data.data,
            data.indices,
            data.indptr,
            data.shape[1],
            data.shape[0])

def max_variance_genes(data, nbins=5, frac=0.2):
    """
    This function identifies the genes that have the max variance
    across a number of bins sorted by mean.

    Args:
        data (array): genes x cells
        nbins (int): number of bins to sort genes by mean expression level. Default: 10.
        frac (float): fraction of genes to return per bin - between 0 and 1. Default: 0.1

    Returns:
        list of gene indices (list of ints)
    """
    # TODO: profile, make more efficient for large matrices
    # 8000 cells: 0.325 seconds
    # top time: sparse.csc_tocsr, csc_matvec, astype, copy, mul_scalar
    # 73233 cells: 5.347 seconds, 4.762 s in sparse_var
    # csc_tocsr: 1.736 s
    # copy: 1.028 s
    # astype: 0.999 s
    # there is almost certainly something superlinear in this method
    # maybe it's to_csr?
    indices = []
    if sparse.issparse(data):
        means, var = sparse_mean_var(data)
    else:
        means = data.mean(1)
        var = data.var(1)
    mean_indices = means.argsort()
    n_elements = int(data.shape[0]/nbins)
    frac_elements = int(n_elements*frac)
    for i in range(nbins):
        bin_i = mean_indices[i*n_elements : (i+1)*n_elements]
        if i==nbins-1:
            bin_i = mean_indices[i*n_elements :]
        var_i = var[bin_i]
        var_sorted = var_i.argsort()
        top_var_indices = var_sorted[len(bin_i) - frac_elements:]
        ind = bin_i[top_var_indices]
        # filter out genes with zero variance
        ind = [index for index in ind if var[index]>0]
        indices.extend(ind)
    return indices

def cell_normalize(data, multiply_means=True):
    """
    Returns the data where the expression is normalized so that the total
    count per cell is equal.

    If multiply_means is true, then the data will be multiplied to have the median UMI count for all cells.
    """
    if sparse.issparse(data):
        data = sparse.csc_matrix(data.astype(float))
        # normalize in-place
        sparse_cell_normalize(data.data,
                data.indices,
                data.indptr,
                data.shape[1],
                data.shape[0],
                multiply_means)
        return data
    data_norm = data.astype(float)
    total_umis = []
    for i in range(data.shape[1]):
        di = data_norm[:,i]
        total_umis.append(di.sum())
        di /= total_umis[i]
    if multiply_means:
        med = np.median(total_umis)
        data_norm *= med
    return data_norm

def log1p(data):
    """
    Returns ln(data+1), whether the original data is dense or sparse.
    """
    if sparse.issparse(data):
        return data.log1p()
    else:
        return np.log1p(data)
