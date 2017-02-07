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
        - data in 2d space
        - list of edges (pairs of cell indices) for each cluster
    """
    # step 1: dimensionality reduction

def pseudotime():
    """
    """
