# dimensionality reduction

import numpy as np
from pois_ll import poisson_dist

def dim_reduce(data, means, weights, d):
    """
    Dimensionality reduction using Poisson distances and MDS.

    Args:
        data (array) - genes x cells
        means (array) - genes x clusters
        weights (array) - clusters x cells
        d (int) - desired dimensionality

    Returns:
        lower-dimensional representation of data
    """
