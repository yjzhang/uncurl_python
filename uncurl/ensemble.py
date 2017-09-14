# ensemble state estimation??

# method based on https://arxiv.org/abs/1702.07186
# combine all the means produced...

import numpy as np
from state_estimation import poisson_estimate_state, initialize_from_assignments

from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, non_negative_factorization
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold

import Cluster_Ensembles as CE

def state_estimation_ensemble(data, k, n_runs=10, M_list=[], **se_params):
    """
    Runs an ensemble method on the list of M results...

    Args:
        data: genes x cells array
        k: number of classes
        n_runs (optional): number of random initializations of state estimation
        M_list (optional): list of M arrays from state estimation
        se_params (optional): optional poisson_estimate_state params

    Returns:
        M_new
        W_new
        ll
    """
    if len(M_list)==0:
        M_list = []
        for i in range(n_runs):
            M, W, ll = poisson_estimate_state(data, k, **se_params)
            M_list.append(M)
    M_stacked = np.hstack(M_list)
    M_new, W_new, ll = poisson_estimate_state(M_stacked, k, **se_params)
    W_new = np.dot(data.T, M_new)
    W_new = W_new/W_new.sum(0)
    return M_new, W_new, ll

def nmf_ensemble(data, k, n_runs=10, W_list=[], **nmf_params):
    """
    Runs an ensemble method on the list of NMF W matrices...

    Args:
        data: genes x cells array (should be log + cell-normalized)
        k: number of classes
        n_runs (optional): number of random initializations of state estimation
        M_list (optional): list of M arrays from state estimation
        se_params (optional): optional poisson_estimate_state params

    Returns:
        W_new
        H_new
    """
    nmf = NMF(k)
    if len(W_list)==0:
        W_list = []
        for i in range(n_runs):
            W = nmf.fit_transform(data)
            W_list.append(W)
    W_stacked = np.hstack(W_list)
    nmf_w = nmf.fit_transform(W_stacked)
    nmf_h = nmf.components_
    H_new = data.T.dot(nmf_w).T
    nmf2 = NMF(k, init='custom')
    nmf_w = nmf2.fit_transform(data, W=nmf_w, H=H_new)
    H_new = nmf2.components_
    #W_new = W_new/W_new.sum(0)
    # alternatively, use nmf_w and h_new as initializations for another NMF round?
    return nmf_w, H_new

def nmf_kfold(data, k, n_runs=10, **nmf_params):
    """
    Runs K-fold ensemble topic modeling (Belford et al. 2017)
    """
    # TODO
    nmf = NMF(k)
    W_list = []
    kf = KFold(n_splits=n_runs, shuffle=True)
    # TODO: randomly divide data into n_runs folds
    for train_index, test_index in kf.split(data.T):
        W = nmf.fit_transform(data[:,train_index])
        W_list.append(W)
    W_stacked = np.hstack(W_list)
    nmf_w = nmf.fit_transform(W_stacked)
    nmf_h = nmf.components_
    H_new = data.T.dot(nmf_w).T
    nmf2 = NMF(k, init='custom')
    nmf_w = nmf2.fit_transform(data, W=nmf_w, H=H_new)
    H_new = nmf2.components_
    #W_new = W_new/W_new.sum(0)
    return nmf_w, H_new

# clustering hack: use consensus clustering to generate a new M and W and use these
# for initialization
# 1. run a bunch of NMFs, get M and W
# 2. run tsne + km on all Ws
# 3. run consensus clustering on km results
# 4. use semi-supervision to get

def nmf_init(data, clusters, k, init='enhanced'):
    """
    runs enhanced NMF initialization from clusterings (Gong 2013)

    There are 3 options for init:
        enhanced - uses EIn-NMF from Gong 2013
        basic - uses means for W, assigns H such that the chosen cluster for a given cell has value 0.75 and all others have 0.25/(k-1).
        nmf - uses means for W, and assigns H using the NMF objective while holding W constant.
    """
    init_w = np.zeros((data.shape[0], k))
    for i in range(k):
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

def nmf_tsne(data, k, n_runs=10, init='enhanced', **params):
    """
    runs tsne-consensus-NMF

    1. run a bunch of NMFs, get W and H
    2. run tsne + km on all WH matrices
    3. run consensus clustering on all km results
    4. use consensus clustering as initialization for a new run of NMF
    5. return the W and H from the resulting NMF run
    """
    clusters = []
    nmf = NMF(k)
    tsne = TSNE(2)
    km = KMeans(k)
    for i in range(n_runs):
        w = nmf.fit_transform(data)
        h = nmf.components_
        tsne_wh = tsne.fit_transform(w.dot(h).T)
        clust = km.fit_predict(tsne_wh)
        clusters.append(clust)
    clusterings = np.vstack(clusters)
    consensus = CE.cluster_ensembles(clusterings, verbose=False, N_clusters_max=k)
    nmf_new = NMF(k, init='custom')
    # TODO: find an initialization for the consensus W and H
    init_w, init_h = nmf_init(data, consensus, k, init)
    W = nmf_new.fit_transform(data, W=init_w, H=init_h)
    H = nmf_new.components_
    return W, H

def poisson_se_tsne(data, k, n_runs=10, init='basic', **se_params):
    """
    runs tsne-consensus-poissonSE
    """
    clusters = []
    tsne = TSNE(2)
    km = KMeans(k)
    for i in range(n_runs):
        m, w, ll = poisson_estimate_state(data, k, **se_params)
        tsne_w = tsne.fit_transform(w.T)
        clust = km.fit_predict(tsne_w)
        clusters.append(clust)
    clusterings = np.vstack(clusters)
    consensus = CE.cluster_ensembles(clusterings, verbose=False, N_clusters_max=k)
    init_m, init_w = nmf_init(data, consensus, k, 'basic')
    M, W, ll = poisson_estimate_state(data, k, init_means=init_m, init_weights=init_w, **se_params)
    return M, W, ll

def lensNMF(data, k, ks=1):
    """
    Runs L-EnsNMF on the data. (Suh et al. 2016)
    """
    # TODO: why is this not working
    n_rounds = k/ks
    R_i = data.copy()
    nmf = NMF(ks)
    nmf2 = NMF(ks, init='custom')
    w_is = []
    h_is = []
    rs = []
    w_i = np.zeros((data.shape[0], ks))
    h_i = np.zeros((ks, data.shape[1]))
    for i in range(n_rounds):
        R_i = R_i - w_i.dot(h_i)
        R_i[R_i < 0] = 0
        """
        P_r = R_i.sum(1)/R_i.sum()
        print P_r.shape
        P_c = R_i.sum(0)/R_i.sum()
        print P_c.shape
        row_choice = np.random.choice(range(len(P_r)), p=P_r)
        print row_choice
        col_choice = np.random.choice(range(len(P_c)), p=P_c)
        print col_choice
        D_r = cosine_similarity(data[row_choice:row_choice+1,:], data)
        D_c = cosine_similarity(data[:,col_choice:col_choice+1].T, data.T)
        D_r = np.diag(D_r.flatten())
        D_c = np.diag(D_c.flatten())
        R_L = D_r.dot(R_i).dot(D_c)
        w_i = nmf.fit_transform(R_L)
        """
        w_i = nmf.fit_transform(R_i)
        h_i = nmf.components_
        #nmf2.fit_transform(R_i, W=w_i, H=nmf.components_)
        #h_i = nmf2.components_
        #h_i[h_i < 0] = 0
        w_is.append(w_i)
        h_is.append(h_i)
        rs.append(R_i)
    return np.hstack(w_is), np.vstack(h_is), rs

