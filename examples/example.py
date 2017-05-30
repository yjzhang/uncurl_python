from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

import uncurl
from uncurl.evaluation import purity

if __name__ == '__main__':
    dat = loadmat('data/SCDE_test.mat')
    data = dat['dat'].toarray()
    centers, assignments = uncurl.kmeans_pp(data, 2)
    lls = uncurl.poisson_ll(data, centers)
    # Poisson clustering
    assignments_poisson, centers = uncurl.poisson_cluster(data, 2, init=centers)
    # NB clustering
    assignments_nb, P, R = uncurl.nb_cluster(data, 2)
    # ZIP clustering
    assignments_zip, M, L = uncurl.zip_cluster(data, 2)
    # State estimation
    means, weights = uncurl.poisson_estimate_state(data, 2, max_iters=5)
    # dimensionality reduction
    X = uncurl.dim_reduce(means, weights, 2)
    proj = np.dot(X, weights)
    true_labs = dat['Lab'][0]
    print 'poisson purity:', purity(assignments_poisson, true_labs, 2)
    print 'NB purity:', purity(assignments_nb, true_labs, 2)
    print 'ZIP purity:', purity(assignments_zip, true_labs, 2)
    # plotting dimensionality reduction
    plt.cla()
    # weight plot
    plt.title('Dimensionality reduction plot - assigned weight labels')
    plt.scatter(proj[0,:], proj[1,:], s=100, cmap='seismic', c=weights[0,:])
    plt.xlabel('dim 1')
    plt.ylabel('dim 2')
    plt.savefig('dat.png')
    plt.cla()
    # Poisson cluster plot
    plt.title('Dimensionality reduction plot - Poisson clustering labels')
    plt.scatter(proj[0,:], proj[1,:], s=100, cmap='seismic', c=assignments_poisson)
    plt.xlabel('dim 1')
    plt.ylabel('dim 2')
    plt.savefig('poisson_cluster_dat.png')
    plt.cla()
    # NB cluster plot
    plt.title('Dimensionality reduction plot - NB clustering labels')
    plt.scatter(proj[0,:], proj[1,:], s=100, cmap='seismic', c=assignments_nb)
    plt.xlabel('dim 1')
    plt.ylabel('dim 2')
    plt.savefig('nb_cluster_dat.png')
    plt.cla()
    # ZIP cluster plot
    plt.title('Dimensionality reduction plot - ZIP clustering labels')
    plt.scatter(proj[0,:], proj[1,:], s=100, cmap='seismic', c=assignments_zip)
    plt.xlabel('dim 1')
    plt.ylabel('dim 2')
    plt.savefig('zip_cluster_dat.png')
    plt.cla()
    # true label plot
    plt.title('Dimensionality reduction plot - true labels')
    plt.scatter(proj[0,:], proj[1,:], cmap='bwr', s=100, alpha=0.7, c=dat['Lab'])
    plt.xlabel('dim 1')
    plt.ylabel('dim 2')
    plt.savefig('labels.png')
