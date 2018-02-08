# cython: linetrace=True 

# utilities for csc matrices...

#import cython
cimport cython

import numpy as np
cimport numpy as np

from libc.math cimport log2

from scipy import sparse
from scipy.special import xlogy

ctypedef fused int2:
    short
    int
    long
    long long

ctypedef fused DTYPE_t:
    float
    double

ctypedef fused numeric:
    short
    unsigned short
    int
    unsigned int
    long
    float
    double

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def sparse_create_libsvm_file(data, str filename):
    """
    Create a libsvm file for use with LightLDA, for sparse matrices.
    """
    cdef int cells = data.shape[1]
    cdef int genes = data.shape[0]
    cdef int[:] indices, indptr
    cdef long[:] data_
    cdef int ind, g, c, start_ind, end_ind
    #cdef int ind, g, c, start_ind, end_ind
    cdef long i
    f = open(filename, "w")
    csc = sparse.csc_matrix(data)
    indices = csc.indices
    indptr = csc.indptr
    data_ = csc.data.astype(np.long)
    strings = []
    for c in range(cells):
        strings.append('1\t')
        start_ind = indptr[c]
        end_ind = indptr[c+1]
        for i2 in range(start_ind, end_ind):
            g = indices[i2]
            i = data_[i2]
            strings.append(str(g)+':'+str(i)+' ')
        strings.append('\n')
    f.write(''.join(strings))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def sparse_create_plda_file(data, str filename):
    """
    Create a file for use with PLDA, for sparse matrices.
    """
    cdef int cells = data.shape[1]
    cdef int genes = data.shape[0]
    cdef int[:] indices, indptr
    cdef long[:] data_
    cdef int ind, g, c, start_ind, end_ind
    #cdef int ind, g, c, start_ind, end_ind
    cdef long i
    f = open(filename, "w")
    csc = sparse.csc_matrix(data)
    indices = csc.indices
    indptr = csc.indptr
    data_ = csc.data.astype(np.long)
    strings = []
    for c in range(cells):
        start_ind = indptr[c]
        end_ind = indptr[c+1]
        for i2 in range(start_ind, end_ind):
            g = indices[i2]
            i = data_[i2]
            strings.append('G' + str(g) + ' ' + str(i) + ' ')
        strings.append('\n')
    f.write(''.join(strings))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def sparse_cell_normalize(np.ndarray[DTYPE_t, ndim=1] data,
        np.ndarray[int2, ndim=1] indices,
        np.ndarray[int2, ndim=1] indptr,
        Py_ssize_t cells,
        Py_ssize_t genes):
    """
    cell_normalize for sparse matrices.
    does cell normalize in place.
    """
    cdef int2 c, start_ind, end_ind, i2
    cdef double s
    cdef double[:] total_umis = np.empty(cells)
    cdef DTYPE_t[:] data_ = data
    for c in range(cells):
        start_ind = indptr[c]
        end_ind = indptr[c+1]
        s = 0
        for i2 in range(start_ind, end_ind):
            s += data_[i2]
        for i2 in range(start_ind, end_ind):
            data_[i2] /= s
        total_umis[c] = s
    cdef double med = np.median(np.asarray(total_umis))
    data *= med

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def sparse_poisson_ll_csc(np.ndarray[numeric, ndim=1] data,
        np.ndarray[int2, ndim=1] indices,
        np.ndarray[int2, ndim=1] indptr,
        int2 genes,
        int2 cells,
        np.ndarray[DTYPE_t, ndim=2] means,
        double eps=1e-10):
    """
    calculates the Poisson log-likelihood for a sparse csc matrix
    and dense array means (genes x k).

    returns a dense matrix of dimension cells x k
    """
    cdef int clusters
    cdef double i, val, v2
    cdef numeric[:] data_ = data
    cdef int2[:] indices_ = indices
    cdef int2[:] indptr_ = indptr
    cdef Py_ssize_t j, c, g, k, ind
    cdef int2 i2
    clusters = means.shape[1]
    cdef double[:,:] ll = np.zeros((cells, clusters), dtype=np.double) - means.sum(0)
    cdef double[:,:] logm = np.log(means+eps)
    for c in range(cells):
        start_ind = indptr_[c]
        end_ind = indptr_[c+1]
        for i2 in range(start_ind, end_ind):
            g = indices_[i2]
            i = data_[i2]
            for k in range(clusters):
                ll[c,k] += i*logm[g, k]
    return np.asarray(ll)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def sparse_means_var_csc(np.ndarray[numeric, ndim=1] data,
        np.ndarray[int2, ndim=1] indices,
        np.ndarray[int2, ndim=1] indptr,
        Py_ssize_t cells,
        Py_ssize_t genes):
    """
    Returns a pair of matrices: mean, variance of a sparse csc matrix.

    Mean/variance is taken for each row.
    """
    cdef int2 c, start_ind, end_ind, i2, g
    cdef double s
    cdef double[:] sq_means = np.zeros(genes)
    cdef double[:] var = np.zeros(genes)
    cdef double[:] means = np.zeros(genes)
    for c in range(cells):
        start_ind = indptr[c]
        end_ind = indptr[c+1]
        for i2 in range(start_ind, end_ind):
            g = indices[i2]
            sq_means[g] += data[i2]**2
            means[g] += data[i2]
    for g in range(genes):
        means[g] = means[g]/cells
        var[g] = sq_means[g]/cells - means[g]**2
    return np.asarray(means), np.asarray(var)

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
    cdef long[:] row = coo.row.astype(np.int64)
    cdef long[:] col = coo.col.astype(np.int64)
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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def symmetric_kld(np.ndarray[DTYPE_t, ndim=1] p1, np.ndarray[DTYPE_t, ndim=1] p2,
        eps=1e-10):
    """
    Gets the symmetric KL divergence between the two points
    (assuming that they're of equal length, and both are probability distributions)

    DOES NOT NORMALIZE THE INPUTS. If the inputs are bad, then the output will be bad.
    """
    cdef int i = 0
    cdef double[:] p1_view = p1
    cdef double[:] p2_view = p2
    cdef double kl1 = 0.0
    for i in range(len(p1)):
        kl1 += p1_view[i]*log2(p1_view[i]/(p2_view[i]+eps))
        kl1 += p2_view[i]*log2(p2_view[i]/(p1_view[i]+eps))
    return kl1

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def jensen_shannon(np.ndarray[DTYPE_t, ndim=1] p1, np.ndarray[DTYPE_t, ndim=1] p2,
        eps=1e-10):
    """
    Jensen-Shannon divergence
    (assuming that they're of equal length, and both are probability distributions)

    DOES NOT NORMALIZE THE INPUTS. If the inputs are bad, then the output will be bad.
    """
    cdef int i = 0
    cdef double[:] p1_view = p1
    cdef double[:] p2_view = p2
    cdef double kl1 = 0.0
    cdef double m
    for i in range(len(p1)):
        m = (p1_view[i]+p2_view[i])/2 + eps
        kl1 += p1_view[i]*log2(p1_view[i]/(m))
        kl1 += p2_view[i]*log2(p2_view[i]/(m))
    return kl1

