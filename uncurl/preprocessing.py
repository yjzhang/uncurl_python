"""
Misc functions...
"""

import numpy as np

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
    var = data.var(1)
    mean_indices = means.argsort()
    n_elements = data.shape[0]/nbins
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


