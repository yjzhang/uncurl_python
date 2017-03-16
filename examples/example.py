from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

import uncurl

if __name__ == '__main__':
    dat = loadmat('data/SCDE_test.mat')
    data = dat['dat'].toarray()
    centers, assignments = uncurl.kmeans_pp(data, 2)
    lls = uncurl.poisson_ll(data, centers)
    # Poisson clustering
    assignments, centers = uncurl.poisson_cluster(data, 2, init=centers)
    # State estimation
    means, weights = uncurl.poisson_estimate_state(data, 2, max_iters=5)
    # dimensionality reduction
    X = uncurl.dim_reduce(means, weights, 2)
    proj = np.dot(X, weights)
    # plotting dimensionality reduction
    plt.cla()
    # weight plot
    plt.title('Dimensionality reduction plot - assigned weight labels')
    plt.scatter(proj[0,:], proj[1,:], s=100, cmap='seismic', c=weights[0,:])
    plt.xlabel('dim 1')
    plt.ylabel('dim 2')
    plt.savefig('dat.png')
    plt.cla()
    # true label plot
    plt.title('Dimensionality reduction plot - true labels')
    plt.scatter(proj[0,:], proj[1,:], cmap='bwr', s=100, alpha=0.7, c=dat['Lab'])
    plt.xlabel('dim 1')
    plt.ylabel('dim 2')
    plt.savefig('labels.png')
