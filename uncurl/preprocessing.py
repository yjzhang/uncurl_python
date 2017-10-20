"""
Misc functions...
"""

import numpy as np
from scipy import sparse

from uncurl.sparse_utils import sparse_cell_normalize

def sparse_var(data):
    """
    Calculates the variance for each row of a sparse matrix.
    """
    data_csr = sparse.csr_matrix(data, dtype=np.float64)
    means = np.array(data_csr.mean(1)).flatten()
    sq = data_csr.power(2)
    means_2 = np.array(sq.mean(1)).flatten()
    var = means_2 - means**2
    return var

def max_variance_genes(data, nbins=10, frac=0.1):
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
    indices = []
    means = data.mean(1)
    if sparse.issparse(data):
        var = sparse_var(data)
        means = np.array(means).flatten()
    else:
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

def cell_normalize(data):
    """
    Returns the data where the expression is normalized so that the total
    count per cell is equal.
    """
    if sparse.issparse(data):
        return sparse_cell_normalize(data)
    data_norm = data.copy().astype(float)
    total_umis = []
    for i in range(data.shape[1]):
        total_umis.append(data_norm[:,i].sum())
        data_norm[:,i] /= total_umis[i]
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
