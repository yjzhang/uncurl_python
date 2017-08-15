# cython: linetrace=True 

# utilities for csc matrices...

#import cython
cimport cython

import numpy as np
cimport numpy as np

from scipy import sparse
from scipy.special import xlogy

DTYPE = np.double
ctypedef np.double_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def sparse_poisson_ll(data, np.ndarray[DTYPE_t, ndim=2] means, eps=1e-10):
    """
    calculates the Poisson log-likelihood for a sparse matrix data (genes x cells)
    and dense array means (genes x k).

    returns a dense matrix of dimension cells x k
    """
    cdef int genes, cells, clusters
    cdef double i, val, v2
    cdef Py_ssize_t j, c, g, k, ind
    genes = data.shape[0]
    cells = data.shape[1]
    clusters = means.shape[1]
    cdef double[:,:] ll = np.zeros((cells, clusters), dtype=np.double) - means.sum(0)
    cdef double[:,:] mv = means
    cdef int[:] row, col
    cdef double[:] data_
    cdef double[:,:] logm = np.log(means+eps)
    # convert to coo format for quicker lookup...
    coo = sparse.coo_matrix(data)
    row = coo.row.astype(np.int32)
    col = coo.col.astype(np.int32)
    data_ = coo.data.astype(np.float64)
    for ind in range(len(data_)):
        i = data_[ind]
        g = row[ind]
        c = col[ind]
        for k in range(clusters):
            ll[c,k] += i*logm[g, k]
    # TODO: should subtract all means even where x is zero as well...
    return np.asarray(ll)

def poisson_dist(np.ndarray[DTYPE_t, ndim=1] p1, np.ndarray[DTYPE_t, ndim=1] p2, eps=1e-10):
    """
    Returns the poisson distance between the two arrays.
    """

def sparse_poisson_dist(p1, np.ndarray[DTYPE_t, ndim=1] p2, eps=1e-10):
    """
    Returns the poisson distance between the two arrays.

    Args:
        p1: 1d sparse column matrix (shape genes x 1),
        p2: 1d ndarray (genes)
    """
    coo = p1
    if not sparse.isspmatrix_coo(coo):
        coo = sparse.coo_matrix(p1)
    cdef int[:] row = coo.row.astype(np.int32)
    cdef int[:] col = coo.col.astype(np.int32)
    cdef double[:] data_ = coo.data.astype(np.float64)
    cdef Py_ssize_t i, g, c

    cdef double[:] logdata = np.log(data_)
    cdef double[:] logp2 = np.log(p2)

    cdef double val, dist
    dist = 0
    cdef double[:] p2_view = p2
    for i in range(len(data_)):
        g = row[i]
        val = data_[i]
        if p2[g]==0:
            pass
        else:
            dist += (val - p2[g])*(logdata[i] - logp2[i])

def poisson_dist_mat(np.ndarray[DTYPE_t, ndim=2] data, np.ndarray[DTYPE_t, ndim=2] centers, eps=1e-10):
    """
    creates a cells x k distance matrix indicating the distances fromm each cell
    to each cluster center.

    Args:
        data: genes x cells
        centers: genes x k
    """
    cdef int genes = data.shape[0]
    cdef int cells = data.shape[1]
    cdef int clusters = centers.shape[1]
    cdef Py_ssize_t i, g, c, k
    cdef double[:,:] dv = data+eps
    cdef double[:,:] cv = centers+eps
    cdef double[:,:] ld = np.log(data+eps)
    cdef double[:,:] lc = np.log(centers+eps)
    cdef double[:,:] distances = np.zeros((cells, clusters))
    for g in range(genes):
        for c in range(cells):
            for k in range(clusters):
                distances[c,k] += (dv[g,c]-cv[g,k])*(ld[g,c]-lc[g,k])
    return np.asarray(distances)

def sparse_poisson_dist_mat(data, np.ndarray[DTYPE_t, ndim=2] centers, eps=1e-10):
    """
    creates a cells x k distance matrix indicating the distances fromm each cell
    to each cluster center.

    Args:
        data: sparse genes x cells
        centers: genes x k
    """
    coo = data
    cdef int genes = data.shape[0]
    cdef int cells = data.shape[1]
    cdef int clusters = centers.shape[1]
    if not sparse.isspmatrix_coo(coo):
        coo = sparse.coo_matrix(data)
    cdef int[:] row = coo.row.astype(np.int32)
    cdef int[:] col = coo.col.astype(np.int32)
    cdef double[:] data_ = coo.data.astype(np.float64)+eps
    cdef double[:] logdata = np.log(data+eps)

    cdef double[:,:] cv = centers+eps
    cdef double[:,:] logc = np.log(centers+eps)

    sparse_centers = sparse.coo_matrix(centers)
    cdef int[:] c_row = sparse_centers.row.astype(np.int32)
    cdef int[:] c_col = sparse_centers.col.astype(np.int32)
    cdef double[:] c_data = sparse_centers.data.astype(np.float64)

    cdef Py_ssize_t i, g, c, k
    cdef double[:,:] distances = np.zeros((cells, clusters))
    for i in range(len(data_)):
        g = row[i]
        c = col[i]
        for k in range(clusters):
            distances[c,k] += (data_[i] - cv[g,k])*(logdata[i] - logc[g,k])
    return np.asarray(distances)
