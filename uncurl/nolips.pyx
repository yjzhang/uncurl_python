# cython: linetrace=True

# cython implementations of optimization functions

# TODO: implement objective/derivative functions for all the distributions,
# nolips
#import cython
cimport cython

from scipy import sparse

import numpy as np
cimport numpy as np
DTYPE = np.double
ctypedef np.double_t DTYPE_t

cdef double eps = 1e-10

M_old = []
M_old.append(np.zeros((10,10)))
mx_cache = {}

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def nolips_update_w(np.ndarray[DTYPE_t, ndim=2] X, np.ndarray[DTYPE_t, ndim=2] M, np.ndarray[DTYPE_t, ndim=2] W, np.ndarray[DTYPE_t, ndim=1] Xsum, disp=False):
    """
    Iteratively runs nolips updates.
    """
    cdef int cells = X.shape[1]
    cdef int genes = X.shape[0]
    cdef int k = W.shape[0]
    cdef int use_mx_cache = False
    if M_old[0].shape[0] == M.shape[0] and M_old[0].shape[1] == M.shape[1]:
        if (M_old[0] == M).all():
            use_mx_cache = True
    if not use_mx_cache:
        M_old[0] = M.copy()
    cdef double[:,:] X_view = X
    cdef double[:,:] M_view = M
    cdef np.ndarray[DTYPE_t, ndim=2] MW = (M.dot(W)+eps).T
    cdef double[:,:] mw_view = MW
    cdef np.ndarray[DTYPE_t, ndim=1] R = M.sum(0)
    cdef double[:] R_view = R
    cdef np.ndarray[DTYPE_t, ndim=1] z = np.zeros(k)
    cdef double lam, ci
    cdef np.ndarray[DTYPE_t, ndim=1] lams = 1/(2*Xsum)
    cdef double[:,:] Wnew_view = np.empty((k, cells), dtype=np.double)
    cdef double[:,:] W_view = W
    cdef np.ndarray[DTYPE_t, ndim=2] y2
    cdef Py_ssize_t i, g, j
    for i in range(cells):
        lam = lams[i]
        if use_mx_cache:
            y2 = mx_cache[i]
        else:
            y2 = np.zeros((genes, k))
            for g in range(genes):
                for j in range(k):
                    y2[g,j] = M_view[g,j]*X_view[g,i]
            y2 = y2.T
            mx_cache[i] = y2
        for j in range(k):
            ci = 0
            for g in range(genes):
                ci += y2[j,g]/mw_view[i,g]
            Wnew_view[j,i] = max(0.0, W_view[j,i]/(1+lam*W_view[j,i]*(R_view[j]-ci)))
    return np.asarray(Wnew_view)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def objective(np.ndarray[DTYPE_t, ndim=2] X, np.ndarray[DTYPE_t, ndim=2] M, np.ndarray[DTYPE_t, ndim=2] W, disp=False):
    """
    Calculates the Poisson Mixture objective value.
    """
    cdef int genes = X.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] d = M.dot(W) + eps
    return np.sum(d - X*np.log(d))/genes

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def cost(np.ndarray[DTYPE_t, ndim=2] X, np.ndarray[DTYPE_t, ndim=2] M, np.ndarray[DTYPE_t, ndim=2] W, disp=False):
    """
    Calculates the log-likelihood of X | M, W
    """
    cdef int genes = X.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] d = M.dot(W) + eps
    return np.sum(d - X*np.log(d))/genes

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
def sparse_nolips_update_w(X, np.ndarray[DTYPE_t, ndim=2] M, np.ndarray[DTYPE_t, ndim=2] W, np.ndarray[DTYPE_t, ndim=1] Xsum, disp=False):
    """
    Iteratively runs nolips updates where X is a sparse matrix.
    """
    cdef int cells = X.shape[1]
    cdef int genes = X.shape[0]
    cdef int k = W.shape[0]
    cdef double[:,:] M_view = M
    cdef np.ndarray[DTYPE_t, ndim=1] R = M.sum(0)
    cdef double[:] R_view = R
    cdef np.ndarray[DTYPE_t, ndim=1] z = np.zeros(k)
    cdef double lam, ci, mw
    cdef np.ndarray[DTYPE_t, ndim=1] lams = 1/(2*Xsum)
    cdef double[:,:] Wnew_view = np.empty((k, cells), dtype=np.double)
    cdef double[:,:] W_view = W
    cdef Py_ssize_t i, g, j, k2, start_ind, end_ind
    # convert to csc
    X_csc = sparse.csc_matrix(X)
    cdef int[:] indices = X_csc.indices
    cdef int[:] indptr = X_csc.indptr
    cdef double[:] data_ = X_csc.data
    cdef double[:,:] cij = np.zeros((cells, k))
    # based on timing results, it seems that parallel w/guided schedule and 4 threads only improves runtime by 10%
    with nogil:
        for i in range(cells):
            start_ind = indptr[i]
            end_ind = indptr[i+1]
            for ind in range(start_ind, end_ind):
                g = indices[ind]
                mw = eps
                for k2 in range(k):
                    mw += M_view[g,k2]*W_view[k2,i]
                mw = data_[ind]/mw
                for j in range(k):
                    cij[i,j] += M_view[g,j]*mw
            lam = lams[i]
            for j in range(k):
                Wnew_view[j,i] = max(0.0, W_view[j,i]/(1+lam*W_view[j,i]*(R_view[j]-cij[i,j])))
    return np.asarray(Wnew_view)


