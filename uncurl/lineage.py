# Lineage tracing and pseudotime calculation

import numpy as np
from dim_reduce import dim_reduce

def lineage(data, means, weights):
    """
    Lineage graph produced by minimum spanning tree

    Args:
        data (array): genes x cells
        means (array): genes x clusters - output of state estimation
        weights (array): clusters x cells - output of state estimation

    Returns:
        - smoothed data in 2d space
        - list of edges (pairs of cell indices) for each cluster
    """
    # step 1: dimensionality reduction
    X = dim_reduce(data, means, weights, 2)
    reduced_data = np.dot(X.T, weights)
    # identifying dominant cell types - max weight for each cell
    genes, cells = data.shape
    clusters = means.shape[1]
    cell_cluster_assignments = []
    for i in range(cells):
        max_cluster = weights[:,i].argmax()
        cell_cluster_assignments.append(max_cluster)
    # fit smooth curve over cell types -5th order fourier series

def pseudotime():
    """
    """
