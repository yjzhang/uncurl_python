# poisson clustering

import numpy as np

from pois_ll import poisson_ll, poisson_dist, zip_ll

def kmeans_pp(data, k, centers=None):
    """
    Generates kmeans++ initial centers.

    Args:
        data (array): A 2d array- genes x cells
        k (int): Number of clusters
        centers (array, optional): if provided, these are one or more known cluster centers. 2d array of genes x number of centers (<=k).

    Returns:
        centers - a genes x k array of cluster means.
        assignments - a cells x 1 array of cluster assignments
    """
    # TODO: what if there is missing data for a given gene?
    # missing data could be if all the entires are -1.
    genes, cells = data.shape
    num_known_centers = 0
    if centers is None:
        centers = np.zeros((genes, k))
    else:
        num_known_centers = centers.shape[1]
        centers = np.concatenate((centers, np.zeros(genes, k-num_known_centers)))
    distances = np.zeros((cells, k))
    distances[:] = np.inf
    if num_known_centers == 0:
        init = np.random.randint(0, cells)
        centers[:,0] = data[:, init]
        num_known_centers+=1
    for c in range(num_known_centers, k):
        for c2 in range(c):
            for i in range(cells):
                distances[i,c2] = poisson_dist(data[:,i], centers[:,c2])
        # choose a new data point as center... probability proportional
        # to distance^2
        min_distances = np.min(distances, 1)
        min_distances = min_distances**2
        min_dist = np.random.choice(range(cells),
                p=min_distances/min_distances.sum())
        centers[:,c] = data[:, min_dist]
    cluster_dists = np.zeros((cells, k))
    for c in range(k):
        cluster_dists[:,c] = np.array([poisson_dist(centers[:,c], data[:,i]) for i in range(cells)])
    new_assignments = np.argmin(cluster_dists, 1)
    return centers, new_assignments

def poisson_cluster(data, k, init=None, max_iters=100):
    """
    Performs Poisson hard EM on the given data.

    Args:
        data (array): A 2d array- genes x cells
        k (int): Number of clusters
        init (array, optional): Initial centers - genes x k array. Default: None, use kmeans++
        max_iters (int, optional): Maximum number of iterations. Default: 100

    Returns:
        a tuple of two arrays: a cells x 1 vector of cluster assignments,
        and a genes x k array of cluster means.
    """
    # TODO: be able to use a combination of fixed and unknown starting points
    # e.g., have init values only for certain genes, have a row of all
    # zeros indicating that kmeans++ should be used for that row.
    genes, cells = data.shape
    init, assignments = kmeans_pp(data, k, centers=init)
    centers = np.copy(init)
    assignments = np.zeros(cells)
    for it in range(max_iters):
        lls = poisson_ll(data, centers)
        #cluster_dists = np.zeros((cells, k))
        # TODO: use log-likelihoods rather than distances
        new_assignments = np.argmax(lls, 1)
        if np.equal(assignments, new_assignments).all():
            return assignments, centers
        for c in range(k):
            centers[:,c] = np.mean(data[:,new_assignments==c], 1)
        assignments = new_assignments
    return assignments, centers

def zip_fit_params(data):
    """
    Returns the ZIP parameters that best fit a given data set.

    Args:
        data (array): 2d array of genes x cells belonging to a given cluster

    Returns:
        L (array): 1d array of means
        M (array): 1d array of zero-inflation parameter
    """
    genes, cells = data.shape
    m = data.mean(1)
    v = data.var(1)
    M = (v-m)/(m**2+v-m)
    M = np.array([min(1, x) for x in M])
    L = (m**2+v-m)/m
    L = np.array([max(0, x) for x in L])
    return L, M

def zip_cluster(data, k, init=None, max_iters=100):
    """
    Performs hard EM clustering using the zero-inflated Poisson distribution.

    Args:
        data (array): A 2d array- genes x cells
        k (int): Number of clusters
        init (array, optional): Initial centers - genes x k array. Default: None, use kmeans++
        max_iters (int, optional): Maximum number of iterations. Default: 100

    Returns:
        assignments (array): integer assignments of cells to clusters (length cells)
        L (array): means (genes x k)
        M (array): zero-inflation parameter (genes x k)
    """
    genes, cells = data.shape
    init, assignments = kmeans_pp(data, k, centers=init)
    centers = np.copy(init)
    M = np.zeros(centers.shape)
    assignments = np.zeros(cells)
    for it in range(max_iters):
        lls = zip_ll(data, centers, M)
        new_assignments = np.argmax(lls, 1)
        if np.equal(assignments, new_assignments).all():
            return assignments, centers, M
        for c in range(k):
            centers[:,c], M[:,c] = zip_fit_params(data[:, new_assignments==c])
        assignments = new_assignments
    return assignments, centers, M
