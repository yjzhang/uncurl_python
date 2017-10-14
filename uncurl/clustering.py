# poisson clustering

import numpy as np
from scipy import sparse
from scipy.optimize import minimize

from .pois_ll import poisson_ll, poisson_dist, zip_ll, zip_ll_row

eps = 1e-8

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
    for c in range(num_known_centers, k):
        c2 = c-1
        # TODO: use different formulation for distance... if sparse, use lls
        # if not sparse, use poisson_dist
        if sparse.issparse(data):
            lls = poisson_ll(data, centers[:,c2:c2+1]).flatten()
            distances[:,c2] = -(lls - lls.max())
            distances[:,c2] /= distances[:,c2].max()
        else:
            for cell in range(cells):
                distances[cell, c2] = poisson_dist(data[:,cell], centers[:,c2])
        # choose a new data point as center... probability proportional
        # to distance^2
        min_distances = np.min(distances, 1)
        min_distances = min_distances**2
        min_dist = np.random.choice(range(cells),
                p=min_distances/min_distances.sum())
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
            return assignments, centers
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
    #M = v/(v+m**2)
    #M[np.isnan(M)] = 0.0
    M = np.array([min(1.0, max(0.0, x)) for x in M])
    L = m + v/m - 1.0
    #L = (v + m**2)/m
    L[np.isnan(L)] = 0.0
    L = np.array([max(0.0, x) for x in L])
    return L, M

def zip_fit_params_mle(data):
    genes, cells = data.shape
    L, M = zip_fit_params(data)
    for i in range(genes):
        result = minimize(zip_ll_row, [L[i], M[i]], args=(data[i,:],),
                bounds=[(eps, None),(0,1)])
        params = result.x
        L[i] = params[0]
        M[i] = params[1]
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
        L (array): Poisson parameter (genes x k)
        M (array): zero-inflation parameter (genes x k)
    """
    genes, cells = data.shape
    init, new_assignments = kmeans_pp(data+eps, k, centers=init)
    centers = np.copy(init)
    M = np.zeros(centers.shape)
    assignments = new_assignments
    for c in range(k):
        centers[:,c], M[:,c] = zip_fit_params_mle(data[:, assignments==c])
    for it in range(max_iters):
        lls = zip_ll(data, centers, M)
        new_assignments = np.argmax(lls, 1)
        if np.equal(assignments, new_assignments).all():
            return assignments, centers, M
        for c in range(k):
            centers[:,c], M[:,c] = zip_fit_params_mle(data[:, assignments==c])
        assignments = new_assignments
    return assignments, centers, M
