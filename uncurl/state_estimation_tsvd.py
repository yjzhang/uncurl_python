# -*- coding: utf-8 -*-
"""
Created on Thu Nov 09 13:20:04 2017

@author: Sumit
"""


from clustering import poisson_cluster
from preprocessing import cell_normalize
from state_estimation import poisson_estimate_state, initialize_from_assignments

import numpy as np

from scipy import sparse

from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, non_negative_factorization, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold

#import Cluster_Ensembles as CE
import ConsensusClustering as CC


def nmf_init(data, clusters, k, init='enhanced'):
    """
    runs enhanced NMF initialization from clusterings (Gong 2013)

    There are 3 options for init:
        enhanced - uses EIn-NMF from Gong 2013
        basic - uses means for W, assigns H such that the chosen cluster for a given cell has value 0.75 and all others have 0.25/(k-1).
        nmf - uses means for W, and assigns H using the NMF objective while holding W constant.
    """
    init_w = np.zeros((data.shape[0], k))
    if sparse.issparse(data):
        for i in range(k):
            if data[:,clusters==i].shape[1]==0:
                point = np.random.randint(0, data.shape[1])
                init_w[:,i] = data[:,point].toarray().flatten()
            else:
                init_w[:,i] = np.array(data[:,clusters==i].mean(1)).flatten()
    else:
        for i in range(k):
            if data[:,clusters==i].shape[1]==0:
                point = np.random.randint(0, data.shape[1])
                init_w[:,i] = data[:,point].flatten()
            else:
                init_w[:,i] = data[:,clusters==i].mean(1)
    init_h = np.zeros((k, data.shape[1]))
    if init == 'enhanced':
        distances = np.zeros((k, data.shape[1]))
        for i in range(k):
            for j in range(data.shape[1]):
                distances[i,j] = np.sqrt(((data[:,j] - init_w[:,i])**2).sum())
        for i in range(k):
            for j in range(data.shape[1]):
                init_h[i,j] = 1/((distances[:,j]/distances[i,j])**(-2)).sum()
    elif init == 'basic':
        init_h = initialize_from_assignments(clusters, k)
    elif init == 'nmf':
        init_h_, _, n_iter = non_negative_factorization(data.T, n_components=k, init='custom', update_H=False, H=init_w.T)
        init_h = init_h_.T
    return init_w, init_h


def poisson_se_multiclust(data, k, n_runs=10, **se_params):
    """
    Initializes state estimation using a consensus of several
    fast clustering/dimensionality reduction algorithms.

    It does a consensus of 8 truncated SVD - k-means rounds, and uses the
    basic nmf_init to create starting points.
    """
    clusters = []
    norm_data = cell_normalize(data)
    if sparse.issparse(data):
        log_data = data.log1p()
        log_norm = norm_data.log1p()
    else:
        log_data = np.log1p(data)
        log_norm = np.log1p(norm_data)
    tsvd_50 = TruncatedSVD(50)
    tsvd_k = TruncatedSVD(k)
    km = KMeans(k)
    tsvd1 = tsvd_50.fit_transform(data.T)
    tsvd2 = tsvd_k.fit_transform(data.T)
    tsvd3 = tsvd_50.fit_transform(log_data.T)
    tsvd4 = tsvd_k.fit_transform(log_data.T)
    tsvd5 = tsvd_50.fit_transform(norm_data.T)
    tsvd6 = tsvd_k.fit_transform(norm_data.T)
    tsvd7 = tsvd_50.fit_transform(log_norm.T)
    tsvd8 = tsvd_k.fit_transform(log_norm.T)
    tsvd_results = [tsvd1, tsvd2, tsvd3, tsvd4, tsvd5, tsvd6, tsvd7, tsvd8]
    clusters = []
    for t in tsvd_results:
        clust = km.fit_predict(t)
        clusters.append(clust)
    
    clusterings = np.vstack(clusters)    
    
    #consensus = CE.cluster_ensembles(clusterings, verbose=False, N_clusters_max=k)
    consensus = CC.PickBestCluster(clusterings)
    init_m, init_w = nmf_init(data, consensus, k, 'basic')
    M, W, ll = poisson_estimate_state(data, k, init_means=init_m, init_weights=init_w, **se_params)
    return M, W, ll