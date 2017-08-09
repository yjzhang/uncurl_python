from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

import uncurl
from uncurl.evaluation import purity

if __name__ == '__main__':
    dat = loadmat('data/GSE60361_dat.mat')
    data = dat['Dat']
    genes = uncurl.max_variance_genes(data, nbins=5, frac=0.2)
    data_subset = data[genes,:]
    k = 7
    centers, assignments = uncurl.kmeans_pp(data_subset, k)
    # Poisson clustering
    assignments_poisson, centers = uncurl.poisson_cluster(data_subset, k, init=centers)
    # NB clustering
    assignments_nb, P, R = uncurl.nb_cluster(data_subset, k)
    # ZIP clustering
    #assignments_zip, M, L = uncurl.zip_cluster(data, k)
    true_labs = dat['ActLabs'][0]
    print 'poisson purity:', purity(assignments_poisson, true_labs)
    print 'NB purity:', purity(assignments_nb, true_labs)
    #print 'ZIP purity:', purity(assignments_zip, true_labs)
    # state estimation
    m, w, ll = uncurl.poisson_estimate_state(data_subset, k, max_iters=10, inner_max_iters=25)
    print 'W purity:', purity(w.argmax(0), true_labs)
    # Robust state estimation
    means, weights, ll, sub_genes = uncurl.robust.robust_estimate_state(data, k, max_iters=10, inner_max_iters=25, disp=True, gene_portion=0.2, use_constant=True)
    print 'W purity:', purity(weights.argmax(0), true_labs)
    means_sub = means[sub_genes, :]
    tsne = TSNE(2)
    MW = m.dot(w)
    tsne_mw = tsne.fit_transform(MW.T)
    km = KMeans(k)
    km_labels = km.fit_predict(tsne_mw)
    km_purity = purity(km_labels, true_labs)
    print 'km_tsne_mw purity:', km_purity
    tsne_w = tsne.fit_transform(w.T)
    km_labels = km.fit_predict(tsne_w)
    km_purity = purity(km_labels, true_labs)
    print 'km_tsne_w purity:', km_purity
    # dimensionality reduction
    X = uncurl.dim_reduce(means, weights, 2)
    proj = np.dot(X.T, weights)
    # plotting dimensionality reduction
    plt.cla()
    # weight plot
    plt.title('Dimensionality reduction plot - assigned weight labels')
    plt.scatter(proj[0,:], proj[1,:], s=100, cmap='seismic', c=weights[0,:])
    plt.xlabel('dim 1')
    plt.ylabel('dim 2')
    plt.savefig('dat2.png')
    plt.cla()
    # Poisson cluster plot
    plt.title('Dimensionality reduction plot - Poisson clustering labels')
    plt.scatter(proj[0,:], proj[1,:], s=100, cmap='seismic', c=assignments_poisson)
    plt.xlabel('dim 1')
    plt.ylabel('dim 2')
    plt.savefig('poisson_cluster_dat2.png')
    plt.cla()
    # NB cluster plot
    plt.title('Dimensionality reduction plot - NB clustering labels')
    plt.scatter(proj[0,:], proj[1,:], s=100, cmap='seismic', c=assignments_nb)
    plt.xlabel('dim 1')
    plt.ylabel('dim 2')
    plt.savefig('nb_cluster_dat2.png')
    plt.cla()
    # ZIP cluster plot
    #plt.title('Dimensionality reduction plot - ZIP clustering labels')
    #plt.scatter(proj[0,:], proj[1,:], s=100, cmap='seismic', c=assignments_zip)
    #plt.xlabel('dim 1')
    #plt.ylabel('dim 2')
    #plt.savefig('zip_cluster_dat2.png')
    #plt.cla()
    # true label plot
    plt.title('Dimensionality reduction plot - true labels')
    plt.scatter(tsne_mw[:,0], tsne_mw[:,1], s=100, c=true_labs)
    plt.xlabel('dim 1')
    plt.ylabel('dim 2')
    plt.savefig('labels_dat2.png')
