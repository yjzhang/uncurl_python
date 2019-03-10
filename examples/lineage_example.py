from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

import uncurl
from uncurl.lineage import fourier_series

if __name__ == '__main__':
    dat = loadmat('data/BranchedSynDat.mat')
    data = dat['Dat'].astype(float)
    # Poisson clustering
    assignments, centers = uncurl.poisson_cluster(data, 3)
    # State estimation
    means, weights, ll = uncurl.run_state_estimation(data, 3)
    #means, weights = np.load('means_weights.npy')
    # dimensionality reduction
    X = uncurl.dim_reduce(means, weights, 2)
    proj = np.dot(X.T, weights)
    cluster_curves, cluster_fitted_vals, cluster_edges, cluster_assignments = uncurl.run_lineage(means, weights, curve_function='poly')
    # dimensionality reduction with true data
    true_weights = dat['X']
    true_means = dat['M']
    X = uncurl.dim_reduce(true_means, true_weights, 2)
    proj_true = np.dot(X.T, true_weights)
    true_curves, true_fitted, true_edges, true_assignments = uncurl.run_lineage(true_means, true_weights)
    # plotting dimensionality reduction, fitted curves
    plt.clf()
    plt.cla()
    plt.title('Dimensionality reduction plot')
    plt.scatter(proj[0,:], proj[1,:], s=30, c=weights.argmax(0), edgecolors='none', alpha=0.7)
    plt.scatter(cluster_fitted_vals[0,:], cluster_fitted_vals[1,:], s=30, c=weights.argmax(0), edgecolors='none', alpha=0.7)
    # connect the lines
    for edge in cluster_edges:
        plt.plot((cluster_fitted_vals[0, edge[0]], cluster_fitted_vals[0, edge[1]]),
                (cluster_fitted_vals[1, edge[0]], cluster_fitted_vals[1, edge[1]]), 'black', linewidth=2)
    plt.xlabel('dim 1')
    plt.ylabel('dim 2')
    plt.savefig('branching_dim_reduce_fitted_poly.png')
    plt.cla()
    # true label plot
    """
    plt.title('Dimensionality reduction plot - true labels')
    plt.scatter(proj[0,:], proj[1,:], cmap='bwr', s=100, alpha=0.7, c=dat['Lab'])
    plt.xlabel('dim 1')
    plt.ylabel('dim 2')
    plt.savefig('labels.png')
    """
