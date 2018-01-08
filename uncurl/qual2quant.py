# Qualitative to Quantitative semi-supervision framework

import numpy as np
from scipy import sparse
import scipy.stats
from sklearn.cluster import KMeans

from .clustering import poisson_cluster

def poisson_test(data1, data2, smoothing=1e-5, return_pval=True):
    """
    Returns a p-value for the ratio of the means of two poisson-distributed datasets.

    Source: http://ncss.wpengine.netdna-cdn.com/wp-content/themes/ncss/pdf/Procedures/PASS/Tests_for_Two_Poisson_Means.pdf

    Gu, K., Ng, H.K.T., Tang, M.L., and Schucany, W. 2008. 'Testing the Ratio of Two Poisson Rates.' Biometrical Journal, 50, 2, 283-298

    Based on W2

    Args:
        data1 (array): 1d array of floats - first distribution
        data2 (array): 1d array of floats - second distribution
        smoothing (float): number to add to each of the datasets
        return_pval (bool): True to return p value; False to return test statistic. Default: True
    """
    data1 = data1.astype(float)
    data2 = data2.astype(float)
    data1 += smoothing
    data2 += smoothing
    X1 = data1.sum()
    X2 = data2.sum()
    N1 = len(data1)
    N2 = len(data2)
    d = float(N1)/float(N2)
    rho = 1.0
    w2 = (X2-X1*(rho/d))/np.sqrt((X2+X1)*(rho/d))
    # return test statistic value (higher indicates that the ratio of data2 to data1 > 1.0)
    if not return_pval:
        return w2
    # return p value (lower indicates that the ratio of data2 to data1 > 1.0)
    return 1.0 - scipy.stats.norm.cdf(w2)

def binarize(qualitative):
    """
    binarizes an expression dataset.
    """
    thresholds = qualitative.min(1) + (qualitative.max(1) - qualitative.min(1))/2.0
    binarized = qualitative > thresholds.reshape((len(thresholds), 1)).repeat(8,1)
    return binarized.astype(int)

def qualNorm_filter_genes(data, qualitative, pval_threshold=0.05, smoothing=1e-5, eps=1e-5):
    """
    Does qualNorm but returns a filtered gene set, based on a p-value threshold.
    """
    genes, cells = data.shape
    clusters = qualitative.shape[1]
    output = np.zeros((genes, clusters))
    missing_indices = []
    genes_included = []
    qual_indices = []
    thresholds = qualitative.min(1) + (qualitative.max(1) - qualitative.min(1))/2.0
    pvals = np.zeros(genes)
    for i in range(genes):
        if qualitative[i,:].max() == -1 and qualitative[i,:].min() == -1:
            missing_indices.append(i)
            continue
        qual_indices.append(i)
        threshold = thresholds[i]
        data_i = data[i,:]
        if sparse.issparse(data):
            data_i = data_i.toarray().flatten()
        assignments, means = poisson_cluster(data_i.reshape((1, cells)), 2)
        means = means.flatten()
        high_i = 1
        low_i = 0
        if means[0]>means[1]:
            high_i = 0
            low_i = 1
        # do a p-value test
        p_val = poisson_test(data_i[assignments==low_i], data_i[assignments==high_i], smoothing=smoothing)
        pvals[i] = p_val
        if p_val <= pval_threshold:
            genes_included.append(i)
        else:
            continue
        high_mean = np.median(data_i[assignments==high_i])
        low_mean = np.median(data_i[assignments==low_i]) + eps
        for k in range(clusters):
            if qualitative[i,k]>threshold:
                output[i,k] = high_mean
            else:
                output[i,k] = low_mean
    output = output[genes_included,:]
    pvals = pvals[genes_included]
    return output, pvals, genes_included

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
    thresholds = qualitative.min(1) + (qualitative.max(1) - qualitative.min(1))/2.0
    for i in range(genes):
        if qualitative[i,:].max() == -1 and qualitative[i,:].min() == -1:
            missing_indices.append(i)
            continue
        qual_indices.append(i)
        threshold = thresholds[i]
        data_i = data[i,:]
        if sparse.issparse(data):
            data_i = data_i.toarray().flatten()
        assignments, means = poisson_cluster(data_i.reshape((1, cells)), 2)
        means = means.flatten()
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
