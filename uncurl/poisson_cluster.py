# poisson clustering...

import numpy as np

from pois_ll import poisson_ll, poisson_dist

def kmeans_pp(data, k):
    """
    Generates kmeans++ initial centers.

    Args:
        data (array): A 2d array- genes x cells
        k (int): Number of clusters

    Returns:
        centers - a genes x k array of cluster means.
    """
    genes, cells = data.shape
    centers = np.zeros((genes, k))
    distances = np.zeros((cells, k))
    distances[:] = np.inf
    centers[:,0] = data[:, np.random.randint(0, cells)]
    for c in range(1,k):
        for c2 in range(c):
            for i in range(cells):
                distances[i,c2] = poisson_dist(data[:,i] - centers[:,c2])
        # choose a new data point as center... probability proportional
        # to distance^2
        min_distances = np.min(distances, 1)
        min_distances = min_distances**2
        min_dist = np.random.choice(range(cells),
                p=min_distances/min_distances.sum())
        centers[:,c] = data[:, min_dist]
    return centers

def poisson_cluster(data, k, init=None, max_iters=1000):
    """
    Performs Poisson hard EM on the given data.

    Args:
        data (array): A 2d array- genes x cells
        k (int): Number of clusters
        init (array): Initial centers - genes x k array. Default: use kmeans++
        max_iters (int): Maximum number of iterations. Default: 1000

    Returns:
        a tuple of two arrays: a cells x 1 vector of cluster assignments,
        and a genes x k array of cluster means.
    """
    # TODO

