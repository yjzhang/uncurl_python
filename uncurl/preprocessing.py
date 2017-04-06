"""
Misc functions...
"""

import numpy as np

def max_variance_genes(data, nbins, frac):
    """
    This function identifies the genes that have the max variance
    across a number of bins sorted by mean.

    Args:
        data (array): genes x cells
        nbins (int): number of bins to sort genes by mean expression level
        frac: fraction of genes to return per bin

    Returns:
        list of gene indices
    """
