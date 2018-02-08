# poisson clustering

import numpy as np
from scipy import sparse

from .pois_ll import poisson_ll, poisson_dist

eps = 1e-10

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
    if sparse.issparse(data) and not sparse.isspmatrix_csc(data):
        data = sparse.csc_matrix(data)
    num_known_centers = 0
    if centers is None:
        centers = np.zeros((genes, k))
    else:
        num_known_centers = centers.shape[1]
        centers = np.concatenate((centers, np.zeros((genes, k-num_known_centers))), 1)
    distances = np.zeros((cells, k))
    distances[:] = np.inf
    if num_known_centers == 0:
        init = np.random.randint(0, cells)
        if sparse.issparse(data):
            centers[:,0] = data[:, init].toarray().flatten()
        else:
            centers[:,0] = data[:, init]
        num_known_centers+=1
    available_cells = list(range(cells))
    for c in range(num_known_centers, k):
        c2 = c-1
        # use different formulation for distance... if sparse, use lls
        # if not sparse, use poisson_dist
        if sparse.issparse(data):
            lls = poisson_ll(data, centers[:,c2:c2+1]).flatten()
            distances[:,c2] = 1 + lls.max() - lls
            distances[:,c2] /= distances[:,c2].max()
        else:
            for cell in range(cells):
                distances[cell, c2] = poisson_dist(data[:,cell], centers[:,c2])
        # choose a new data point as center... probability proportional
        # to distance^2
        min_distances = np.min(distances, 1)
        min_distances = min_distances**2
        min_distances = min_distances[available_cells]
        # should be sampling without replacement
        min_dist = np.random.choice(available_cells,
                p=min_distances/min_distances.sum())
        available_cells.pop(available_cells.index(min_dist))
        if sparse.issparse(data):
            centers[:,c] = data[:, min_dist].toarray().flatten()
        else:
            centers[:,c] = data[:, min_dist]
    lls = poisson_ll(data, centers)
    new_assignments = np.argmax(lls, 1)
    centers[centers==0.0] = eps
    return centers, new_assignments

def poisson_cluster(data, k, init=None, max_iters=100):
    """
    Performs Poisson hard EM on the given data.

    Args:
        data (array): A 2d array- genes x cells. Can be dense or sparse; for best performance, sparse matrices should be in CSC format.
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
    #print 'starting: ', centers
    if sparse.issparse(data) and not sparse.isspmatrix_csc(data):
        data = sparse.csc_matrix(data)
    init, assignments = kmeans_pp(data, k, centers=init)
    centers = np.copy(init)
    assignments = np.zeros(cells)
    for it in range(max_iters):
        lls = poisson_ll(data, centers)
        #cluster_dists = np.zeros((cells, k))
        new_assignments = np.argmax(lls, 1)
        if np.equal(assignments, new_assignments).all():
            #print 'ending: ', centers
            return new_assignments, centers
        for c in range(k):
            if sparse.issparse(data):
                if data[:,new_assignments==c].shape[0]==0:
                    # re-initialize centers?
                    new_c, _ = kmeans_pp(data, k, centers[:,:c])
                    centers[:,c] = new_c[:,c]
                else:
                    centers[:,c] = np.asarray(data[:,new_assignments==c].mean(1)).flatten()
            else:
                if len(data[:,new_assignments==c])==0:
                    new_c, _ = kmeans_pp(data, k, centers[:,:c])
                    centers[:,c] = new_c[:,c]
                else:
                    centers[:,c] = np.mean(data[:,new_assignments==c], 1)
        assignments = new_assignments
    return assignments, centers

