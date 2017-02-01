# Qualitative to Quantitative semi-supervision framework

import numpy as np

from poisson_cluster import poisson_cluster

def qual2quant(data, binarized):
    """
    Generates starting points using binarized data.

    Args:
        data (array): 2d array of genes x cells
        binarized (array): binary data - genes x clusters

    Returns:
        Array of starting positions - genes x clusters
    """
    # cluster the genes
    genes, cells = data.shape
    clusters = binarized.shape[1]
    output = np.zeros((genes, clusters))
    for i in range(genes):
        assignments, means = poisson_cluster(data[i,:].reshape((1, cells)))
        high_mean = max(means)
        low_mean = min(means)
        for k in range(clusters):
            if binarized[i,k]>0:
                output[i,k] = high_mean
            else:
                output[i,k] = low_mean
    return output
