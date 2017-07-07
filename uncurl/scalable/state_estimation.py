"""
State estimation using SGD

(how to do it?)

TODO: be able to use sparse (CSC) matrices

Basically, we observe one (cell_id, gene_level) pair at a time, iterating
through the data point by point, updating the gradient based on that point.

"""

import random

import numpy as np
from scipy.sparse import issparse


def m_grad(m, X, w):
    """
    """
    pass

def w_grad(w, X, m):
    """
    """
    pass

def cost_grad(th, Xr, X, n):
    """
    translated from the matlab
    """
    xth = np.dot(X, th)
    cost = (1./n)*sum(xth - Xr*np.log(xth))
    temp = (Xr/xth)
    grad  = (1./n)*(1-temp)*X.T
    return cost, grad


def poisson_estimate_state(data, clusters, init_means=None, init_weights=None, max_iters=10, tol=1e-4, eta=1e-4,  disp=True):
    """
    Runs Poisson state estimation on a sparse data matrix...
    """
    # If data is a sparse (CSC) matrix: loop through points
    # otherwise, loop
    genes, cells = data.shape
    W = np.random.random((clusters, cells))
    M = np.random.random((genes, clusters))
    if issparse(data):
        points = zip(data.nonzero())
        for i in range(max_iters):
            random.shuffle(points)
            # 1. estimate W
            for p1, p2 in points:
                x = data[p1,p2]
                cost, grad = cost_grad(W[:,p2], x, M[p1,:], 1)
            # 2. estimate M
    else:
        print('Warning: data is not sparse')
