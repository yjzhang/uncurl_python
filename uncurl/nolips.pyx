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
    cdef np.ndarray[DTYPE_t, ndim=2] d = M.dot(W)
    return np.sum(d - X*np.log(d))/genes

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def sparse_objective(X, np.ndarray[DTYPE_t, ndim=2] M, np.ndarray[DTYPE_t, ndim=2] W, disp=False):
    """
    Calculates the Poisson Mixture objective value for a sparse matrix x.
    """
    # TODO: I'm not sure how to calculate this efficiently so it's currently zero'ed out.
    return 0
    cdef int cells = X.shape[1]
    cdef int genes = X.shape[0]
    cdef int clusters = W.shape[0]
    cdef double i, d
    cdef Py_ssize_t j, k
    cdef double obj = 0
    coo = sparse.coo_matrix(X).astype(np.double)
    for i, g, c in zip(coo.data, coo.row, coo.col):
        for k in range(clusters):
            d = np.sum(M[g,:]*W[:,c])
            obj += d - i*np.log(d)
    return obj/genes

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
    # what if doing M*W causes memory issues?
    #cdef np.ndarray[DTYPE_t, ndim=2] MW = (M.dot(W)+eps).T
    #cdef double[:,:] mw_view = MW
    cdef double[:,:] M_view = M
    cdef np.ndarray[DTYPE_t, ndim=1] R = M.sum(0)
    cdef double[:] R_view = R
    cdef np.ndarray[DTYPE_t, ndim=1] z = np.zeros(k)
    cdef double lam, ci, mw
    cdef np.ndarray[DTYPE_t, ndim=1] lams = 1/(2*Xsum)
    cdef double[:,:] Wnew_view = np.empty((k, cells), dtype=np.double)
    cdef double[:,:] W_view = W
    #cdef np.ndarray[DTYPE_t, ndim=1] xrow
    #cdef np.ndarray[DTYPE_t, ndim=2] y2
    cdef Py_ssize_t i, g, j, k2
    #X = sparse.csc_matrix(X)
    # convert to coo
    X_coo = sparse.coo_matrix(X)
    cdef int[:] row, col
    cdef double[:] data_
    row = X_coo.row#.astype(np.int32)
    col = X_coo.col#.astype(np.int32)
    data_ = X_coo.data#.astype(np.float64)
    cdef double[:,:] cij = np.zeros((cells, k))
    for ind in range(len(data_)):
        i = col[ind]
        g = row[ind]
        xig = data_[ind]
        mw = eps
        for k2 in range(k):
            mw += M_view[g,k2]*W_view[k2,i]
        mw = xig/mw
        for j in range(k):
            cij[i,j] += M_view[g,j]*mw
    for i in range(cells):
        lam = lams[i]
        for j in range(k):
            Wnew_view[j,i] = max(0.0, W_view[j,i]/(1+lam*W_view[j,i]*(R_view[j]-cij[i,j])))
    return np.asarray(Wnew_view)



