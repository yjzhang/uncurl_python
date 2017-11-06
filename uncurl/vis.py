# basic functions for visualization of clustering, state estimation, lineage

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_poisson_w(w, labels, filename, method='pca', figsize=(18,10), title='', **scatter_options):
    """
    Saves a scatter plot of a visualization of W, the result from Poisson SE.
    """
    if method == 'pca':
        pca = PCA(2)
        r_dim_red = pca.fit_transform(w.T).T
    elif method == 'tsne':
        pass
    else:
        print("Method is not available. use 'pca' (default) or 'tsne'.")
        return
    visualize_dim_red(r_dim_red, labels, filename, figsize, title, **scatter_options)

def visualize_dim_red(r, labels, filename, figsize=(18,10), title='', **scatter_options):
    """
    Saves a scatter plot of a (2,n) matrix r, where each column is a cell.
    """
    plt.figure(figsize=figsize)
    plt.cla()
    for i in set(labels):
        plt.scatter(r[0, labels==i], r[1, labels==i], label=i, **scatter_options)
    plt.title(title)
    plt.legend()
    plt.savefig(filename, dpi=100)
