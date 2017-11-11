import numpy as np
from scipy import sparse
from scipy.optimize import minimize

from .clustering import kmeans_pp
from .zip_utils import zip_ll, zip_ll_row

eps = 1e-8


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

