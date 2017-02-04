from scipy.io import loadmat
import numpy as np

import uncurl

if __name__ == '__main__':
    dat = loadmat('data/SCDE_k2_sup.mat')
    data = dat['Dat']
    centers = uncurl.kmeans_pp(data, 3)
    lls = uncurl.poisson_ll(data, centers)
    # Poisson clustering
    assignments, centers = uncurl.poisson_cluster(data, 3, init=centers)
    # State estimation
    means, weights = uncurl.poisson_estimate_state(data, centers, max_iters=5)
    # dimensionality reduction
    X = uncurl.dim_reduce(data, means, weights, 2)
    proj = np.dot(X.transpose(), weights)
