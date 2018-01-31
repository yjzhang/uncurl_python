# parallel sparse implementation of nolips Poisson optimization

#import cython
cimport cython

from cython.parallel import prange

from scipy import sparse

import numpy as np
cimport numpy as np
#DTYPE = np.double
#ctypedef np.double_t DTYPE_t

# TODO: use fused types
ctypedef fused int2:
    short
    int
    long
    long long

ctypedef fused DTYPE_t:
    float
    double

cdef double eps = 1e-10

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef inline void _update(int2 i, DTYPE_t[:] data_, int2[:] indices, int2[:] indptr, DTYPE_t[:,:] cij, double[:] R_view, double[:,:] M_view, double[:,:] W_view, double[:,:] Wnew_view, double lam, double eps, int2 k) nogil:
    # NoLips in-place update for a single cell/column of w.
    # all these updates can run in parallel.
    cdef int2 start_ind = indptr[i]
    cdef int2 end_ind = indptr[i+1]
    cdef int2 g, k2, j, ind
    cdef double mw
    for ind in range(start_ind, end_ind):
        g = indices[ind]
        mw = eps
        for k2 in range(k):
            mw += M_view[g,k2]*W_view[k2,i]
        mw = data_[ind]/mw
        for j in range(k):
            cij[i,j] += M_view[g,j]*mw
    for j in range(k):
        Wnew_view[j,i] = max(0.0, W_view[j,i]/(1+lam*W_view[j,i]*(R_view[j]-cij[i,j])))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def sparse_nolips_update_w(np.ndarray[DTYPE_t, ndim=1] X_data,
        np.ndarray[int2, ndim=1] X_indices,
        np.ndarray[int2, ndim=1] X_indptr,
        int2 cells,
        int2 genes,
        np.ndarray[DTYPE_t, ndim=2] M,
        np.ndarray[DTYPE_t, ndim=2] W,
        np.ndarray[DTYPE_t, ndim=1] Xsum, int2 n_threads = 4, disp=False):
    """
    Parallel nolips...

    Args:
        X (csc sparse array): data with shape genes x cells
        M (array): genes x k
        W (array): k x cells
        Xsum (array): X.sum(0) - sum each column of X - has length cells
        n_threads (int2): number of threads
        disp (bool): currently unused

    Returns:
        Updated copy of W
    """
    cdef int2 k = W.shape[0]
    cdef double[:,:] M_view = M
    cdef np.ndarray[DTYPE_t, ndim=1] R = M.sum(0)
    cdef double[:] mw_view
    cdef double[:] R_view = R
    cdef np.ndarray[DTYPE_t, ndim=1] z = np.zeros(k)
    cdef double lam, mw, xig
    cdef np.ndarray[DTYPE_t, ndim=1] lams = 1/(2*Xsum)
    cdef double[:] lams_view = lams
    cdef double[:,:] Wnew_view = np.empty((k, cells), dtype=np.double)
    cdef double[:,:] W_view = W
    cdef Py_ssize_t i
    #X_csc = sparse.csc_matrix(X)
    # when there are more than 2^31 elements, will be long.
    # so this function won't work - have to deal with this in the calling
    # function.
    #cdef int2[:] indices, indptr
    cdef int2[:] indices = X_indices
    cdef int2[:] indptr = X_indptr
    cdef DTYPE_t[:] data_ = X_data
    cdef DTYPE_t[:,:] cij = np.zeros((cells, k))
    # schedules: guided, 
    for i in prange(cells, schedule="guided", nogil=True, num_threads=n_threads):
        _update(i, data_, indices, indptr, cij, R_view, M_view, W_view, Wnew_view, lams_view[i], eps, k)
    return np.asarray(Wnew_view)

