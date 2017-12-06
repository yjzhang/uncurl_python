# Qualitative to Quantitative semi-supervision framework

import numpy as np
from scipy import sparse
import scipy.stats
from sklearn.cluster import KMeans

from .clustering import poisson_cluster

def find_bimodal(data, p_genes):
    """
    finds putatively bimodal genes
    """

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
        data_i = data[i,:]
        if sparse.issparse(data):
            data_i = data_i.toarray().flatten()
        assignments, means = poisson_cluster(data_i.reshape((1, cells)), 2)
        means = means.flatten()
        high_mean = means.max()
        low_mean = means.min()
        high_i = 1
        low_i = 0
        if means[0]>means[1]:
            high_i = 0
            low_i = 1
        high_mean = np.median(data_i[assignments==high_i])
        low_mean = np.median(data_i[assignments==low_i])
        for k in range(clusters):
            if qualitative[i,k]>threshold:
                output[i,k] = high_mean
            else:
                output[i,k] = low_mean
    if missing_indices:
        assignments, means = poisson_cluster(data[qual_indices, :], clusters, output[qual_indices, :], max_iters=1)
        for ind in missing_indices:
            for k in range(clusters):
                if len(assignments==k)==0:
                    output[ind, k] = data[ind,:].mean()
                else:
                    output[ind, k] = data[ind, assignments==k].mean()
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
    # TODO: assign to closest
    return output
