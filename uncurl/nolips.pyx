
# cython implementations of optimization functions

# TODO: implement objective/derivative functions for all the distributions,
# nolips
import cython
cimport cython


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
    cells = X.shape[1]
    genes = X.shape[0]
    k = W.shape[0]
    use_mx_cache = False
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
    cdef np.ndarray[DTYPE_t, ndim=2] W_new = np.zeros((k, cells))
    cdef double[:,:] W_view = W
    cdef np.ndarray[DTYPE_t, ndim=2] y2
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
            W_new[j,i] = max(0.0, W_view[j,i]/(1+lam*W_view[j,i]*(R_view[j]-ci)))
    return W_new

