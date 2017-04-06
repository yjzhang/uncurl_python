# Qualitative to Quantitative semi-supervision framework

import numpy as np

from poisson_cluster import poisson_cluster

def qual2quant(data, qualitative):
    """
    Generates starting points using binarized data.

    Args:
        data (array): 2d array of genes x cells
        qualitative (array): 2d array of numerical data - genes x clusters

    Returns:
        Array of starting positions for state estimation or
        clustering, with shape genes x clusters
    """
    # TODO: if 'binarized' is not binary... convert it to binary by setting
    # the threshhold at the midpoint of the range for each gene
    # cluster the genes
    genes, cells = data.shape
    clusters = qualitative.shape[1]
    output = np.zeros((genes, clusters))
    for i in range(genes):
        threshold = (qualitative[i,:].max() - qualitative[i,:].min())/2.0
        assignments, means = poisson_cluster(data[i,:].reshape((1, cells)))
        high_mean = max(means)
        low_mean = min(means)
        for k in range(clusters):
            if qualitative[i,k]>threshold:
                output[i,k] = high_mean
            else:
                output[i,k] = low_mean
    return output
