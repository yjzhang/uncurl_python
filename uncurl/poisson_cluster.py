# poisson clustering...

import numpy as np

def kmeans_pp(data, k):
    """
    Generates kmeans++ initial centers.

    Args:
        data (array): A 2d array- genes x cells
        k (int): Number of clusters

    Returns:
        a tuple of two arrays: a cells x 1 array of cluster assignments,
        and a genes x k array of cluster means.
    """

def poisson_cluster(data, k, max_iters=1000):
    """
    Performs Poisson hard EM on the given data.

    Args:
        data (array): A 2d array- genes x cells
        k (int): Number of clusters
        max_iters (int): Maximum number of iterations

    Returns:
        a tuple of two arrays: a cells x 1 array of cluster assignments,
        and a genes x k array of cluster means.
    """


