# ensemble state estimation??

# method based on https://arxiv.org/abs/1702.07186
# combine all the means produced...

import numpy as np
from state_estimation import poisson_estimate_state

from sklearn.decomposition import NMF

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

def nmf_ensemble(data, k, n_runs=10, W_list=[], **se_params):
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
    H_new = data.T.dot(nmf_h)
    #W_new = W_new/W_new.sum(0)
    return nmf_w, H_new

def state_estimation_kfold(data, k, **se_params):
    """
    Runs K-fold ensemble topic modeling
    """
    # TODO
