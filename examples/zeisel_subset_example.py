from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.manifold import TSNE

import uncurl
from uncurl.sparse_utils import symmetric_kld
from uncurl.vis import visualize_dim_red

# note: this whole script should finish in under a few minutes.

if __name__ == '__main__':

    # 1. load data - 753 cells, 19971 genes
    dat = loadmat('data/GSE60361_dat.mat')
    data = dat['Dat']
    true_labels = dat['ActLabs'].flatten()
    data_csc = sparse.csc_matrix(data)

    # 2. gene selection
    genes = uncurl.max_variance_genes(data_csc, nbins=5, frac=0.2)
    data_subset = data_csc[genes,:]

    # 3. state estimation
    k = 7 # number of clusters to use
    M, W, ll = uncurl.poisson_estimate_state(data_subset, k)
    argmax_labels = W.argmax(0)

    # 4. visualization

    # mds visualization
    mds_proj = uncurl.mds(M, W, 2)
    visualize_dim_red(mds_proj, true_labels, 'GSE60361_mds_true_labels.png', title='MDS', figsize=(12,7), alpha=0.5)

    # tsne visualization
    tsne = TSNE(2, metric=symmetric_kld)
    tsne_w = tsne.fit_transform(W.T)
    # plot using true labels
    visualize_dim_red(tsne_w.T, true_labels, 'GSE60361_tsne_true_labels.png', title='TSNE(W)', figsize=(12,7), alpha=0.5)
    # plot using assigned labels
    visualize_dim_red(tsne_w.T, argmax_labels, 'GSE60361_tsne_argmax_labels.png', title='TSNE(W)', figsize=(12,7), alpha=0.5)
