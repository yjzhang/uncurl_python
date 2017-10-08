# parallel sparse implementation of nolips Poisson optimization

#import cython
cimport cython

from cython.parallel import prange

from scipy import sparse

import numpy as np
cimport numpy as np
DTYPE = np.double
ctypedef np.double_t DTYPE_t

cdef double eps = 1e-10

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def sparse_objective(X, np.ndarray[DTYPE_t, ndim=2] M, np.ndarray[DTYPE_t, ndim=2] W, disp=False):
    """
    Calculates the Poisson Mixture objective value for a sparse matrix x.
    """
    cdef int cells = X.shape[1]
    cdef int genes = X.shape[0]
    cdef int clusters = W.shape[0]
    cdef double d
    cdef Py_ssize_t i, ind, g, c, start_ind, end_ind
    cdef double obj = 0
    # use a csc matrix, iterate through 
    X_csc = sparse.csc_matrix(X)
    cdef int[:] indices, indptr
    indices = X_csc.indices
    indptr = X_csc.indptr
    cdef double[:] data_ = X_csc.data
    cdef double[:] mw = np.zeros(len(data_))
    with nogil:
        for i in range(cells):
            c = i
            start_ind = indptr[i]
            end_ind = indptr[i+1]
            for ind in range(start_ind, end_ind):
                g = indices[ind]
                d = 0
                for k in range(clusters):
                    d += M[g,k]*W[k,c]
                mw[ind] = d
    cdef np.ndarray[DTYPE_t, ndim=1] D = np.asarray(mw)
    cdef np.ndarray[DTYPE_t, ndim=1] data = np.asarray(data_)
    obj = np.sum(-data*np.log(D))
    M_sparse = sparse.csr_matrix(M)
    W_sparse = sparse.csr_matrix(W)
    MW_sparse = M_sparse*W_sparse
    obj += MW_sparse.sum()
    return obj/genes

def cost(X, np.ndarray[DTYPE_t, ndim=2] M, np.ndarray[DTYPE_t, ndim=2] W, disp=False):
    """
    Calculates the log likelihood of X | M, W, where X is sparse
    """
    cdef double ll0 = sparse_objective(X, M, W)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef inline void _update(int i, double[:] data_, int[:] indices, int[:] indptr, double[:,:] cij, double[:] R_view, double[:,:] M_view, double[:,:] W_view, double[:,:] Wnew_view, double lam, float eps, int k) nogil:
    cdef int start_ind = indptr[i]
    cdef int end_ind = indptr[i+1]
    cdef int g, k2, j
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
def sparse_nolips_update_w(X, np.ndarray[DTYPE_t, ndim=2] M, np.ndarray[DTYPE_t, ndim=2] W, np.ndarray[DTYPE_t, ndim=1] Xsum, int n_threads = 4, disp=False):
    """
    Parallel nolips...
    """
    cdef int cells = X.shape[1]
    cdef int genes = X.shape[0]
    cdef int k = W.shape[0]
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
    #X_csc = X
    X_csc = sparse.csc_matrix(X)
    cdef int[:] indices, indptr
    indices = X_csc.indices
    indptr = X_csc.indptr
    cdef double[:] data_
    data_ = X_csc.data
    cdef double[:,:] cij = np.zeros((cells, k))
    cdef int size = len(data_)
    cdef int start_ind, end_ind
    # based on timing results, it seems that parallel w/guided schedule and 4 threads only improves runtime by 10%
    for i in prange(cells, schedule="guided", nogil=True, num_threads=n_threads):
        _update(i, data_, indices, indptr, cij, R_view, M_view, W_view, Wnew_view, lams[i], eps, k)
    return np.asarray(Wnew_view)

