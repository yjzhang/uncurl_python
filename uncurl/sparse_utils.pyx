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
    cdef double[:,:] ll = np.zeros((cells, clusters), dtype=np.double)
    cdef double[:,:] mv = means
    cdef int[:] row, col
    cdef double[:] data_
    cdef double[:,:] logm = np.log(means)
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
            ll[c,k] += i*logm[g, k] - mv[g, k]
    return np.asarray(ll)

def poisson_distances(np.ndarray[DTYPE_t, ndim=1] p1, np.ndarray[DTYPE_t, ndim=1] p2, eps=1e-10):
    """
    Returns a 
    """
