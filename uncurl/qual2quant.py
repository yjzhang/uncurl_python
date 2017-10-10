# Qualitative to Quantitative semi-supervision framework

import numpy as np
from sklearn.cluster import KMeans

from clustering import poisson_cluster

def qualNorm(data, qualitative):
    """
    Generates starting points using binarized data. If qualitative data is missing for a given gene, all of its entries should be -1 in the qualitative matrix.

    Args:
        data (array): 2d array of genes x cells
        qualitative (array): 2d array of numerical data - genes x clusters

    Returns:
        Array of starting positions for state estimation or
        clustering, with shape genes x clusters
    """
    # TODO: if 'binarized' is not binary... convert it to binary by setting
    # the threshhold at the midpoint of the range for each gene
    # cluster the genes
    genes, cells = data.shape
    clusters = qualitative.shape[1]
    output = np.zeros((genes, clusters))
    missing_indices = []
    qual_indices = []
    for i in range(genes):
        if qualitative[i,:].max() == -1 and qualitative[i,:].min() == -1:
            missing_indices.append(i)
            continue
        qual_indices.append(i)
        threshold = (qualitative[i,:].max() - qualitative[i,:].min())/2.0
        assignments, means = poisson_cluster(data[i,:].reshape((1, cells)), 2)
        high_mean = means.max()
        low_mean = means.min()
        for k in range(clusters):
            if qualitative[i,k]>threshold:
                output[i,k] = high_mean
            else:
                output[i,k] = low_mean
    if missing_indices:
        assignments, means = poisson_cluster(data[qual_indices, :], clusters, output[qual_indices, :], max_iters=1)
        for ind in missing_indices:
            for k in range(clusters):
                output[ind, k] = np.mean(data[ind, assignments==k])
    return output


def qualNormGaussian(data, qualitative):
    """
    Generates starting points using binarized data. If qualitative data is missing for a given gene, all of its entries should be -1 in the qualitative matrix.

    Args:
        data (array): 2d array of genes x cells
        qualitative (array): 2d array of numerical data - genes x clusters

    Returns:
        Array of starting positions for state estimation or
        clustering, with shape genes x clusters
    """
    genes, cells = data.shape
    clusters = qualitative.shape[1]
    output = np.zeros((genes, clusters))
    missing_indices = []
    qual_indices = []
    for i in range(genes):
        if qualitative[i,:].max() == -1 and qualitative[i,:].min() == -1:
            missing_indices.append(i)
            continue
        qual_indices.append(i)
        threshold = (qualitative[i,:].max() - qualitative[i,:].min())/2.0
        kmeans = KMeans(n_clusters = 2).fit(data[i,:].reshape((1, cells)))
        assignments = kmeans.labels_
        means = kmeans.cluster_centers_
        high_mean = means.max()
        low_mean = means.min()
        for k in range(clusters):
            if qualitative[i,k]>threshold:
                output[i,k] = high_mean
            else:
                output[i,k] = low_mean
    if missing_indices:
        #generating centers for missing indices 
        M_init = output[qual_indices, :]
        kmeans = KMeans(n_clusters = 2, init = M_init, max_iter = 1).fit(data[qual_indices, :])
        assignments = kmeans.labels_
        #assignments, means = poisson_cluster(data[qual_indices, :], clusters, output[qual_indices, :], max_iters=1)
        for ind in missing_indices:
            for k in range(clusters):
                output[ind, k] = np.mean(data[ind, assignments==k])
    return output
