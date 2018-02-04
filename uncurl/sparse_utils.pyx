# cython: linetrace=True 

# utilities for csc matrices...

#import cython
cimport cython

import numpy as np
cimport numpy as np

from libc.math cimport log2

from scipy import sparse
from scipy.special import xlogy

DTYPE = np.double
ctypedef np.double_t DTYPE_t

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
def sparse_cell_normalize(data):
    """
    cell_normalize for sparse matrices
    """
    cdef int cells = data.shape[1]
    cdef int genes = data.shape[0]
    cdef int[:] indices, indptr
    cdef double[:] data_
    cdef int ind, g, c, start_ind, end_ind
    #cdef int ind, g, c, start_ind, end_ind
    cdef double i, s
    csc = sparse.csc_matrix(data)
    if csc.indptr.dtype == np.int64:
        return sparse_cell_normalize_long(data)
    csc_new = csc.copy().astype(np.double)
    indices = csc_new.indices
    indptr = csc_new.indptr
    data_ = csc_new.data
    cdef double[:] total_umis = np.empty(cells)
    for c in range(cells):
        start_ind = indptr[c]
        end_ind = indptr[c+1]
        s = 0
        for i2 in range(start_ind, end_ind):
            i = data_[i2]
            s += i
        for i2 in range(start_ind, end_ind):
            data_[i2] /= s
        total_umis[c] = s
    med = np.median(np.asarray(total_umis))
    csc_new *= med
    return csc_new

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def sparse_cell_normalize_long(data):
    """
    cell_normalize for sparse matrices
    """
    cdef Py_ssize_t cells = data.shape[1]
    cdef Py_ssize_t genes = data.shape[0]
    cdef long[:] indices, indptr
    cdef double[:] data_
    cdef Py_ssize_t ind, g, c, start_ind, end_ind
    cdef long i2
    #cdef int ind, g, c, start_ind, end_ind
    cdef double i, s
    csc = sparse.csc_matrix(data)
    csc_new = csc.copy().astype(np.double)
    indices = csc_new.indices.astype(np.int64)
    indptr = csc_new.indptr.astype(np.int64)
    data_ = csc_new.data
    cdef double[:] total_umis = np.empty(cells)
    for c in range(cells):
        start_ind = indptr[c]
        end_ind = indptr[c+1]
        s = 0
        for i2 in range(start_ind, end_ind):
            i = data_[i2]
            s += i
        for i2 in range(start_ind, end_ind):
            data_[i2] /= s
        total_umis[c] = s
    med = np.median(np.asarray(total_umis))
    csc_new *= med
    return csc_new


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def sparse_poisson_ll(data, np.ndarray[DTYPE_t, ndim=2] means, eps=1e-10):
    """
    calculates the Poisson log-likelihood for a sparse matrix data (genes x cells)
    and dense array means (genes x k).

    returns a dense matrix of dimension cells x k
    """
    # TODO: this does not deal with longs. maybe we need a more efficient way?
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
    row = coo.row
    col = coo.col
    data_ = coo.data.astype(np.float64)
    for ind in range(len(data_)):
        i = data_[ind]
        g = row[ind]
        c = col[ind]
        for k in range(clusters):
            ll[c,k] += i*logm[g, k]
    return np.asarray(ll)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def sparse_var(data):
    """
    Calculates the variance along each row of a sparse matrix.
    """
    # TODO: is this really necessary?
    data_csr = sparse.csr_matrix(data)
    cdef double[:] means = np.array(data.mean(1)).flatten()

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

