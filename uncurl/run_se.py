# state estimation with poisson convex mixture model

from .state_estimation import poisson_estimate_state
from .nb_state_estimation import nb_estimate_state
from .zip_state_estimation import zip_estimate_state
from .nmf_wrapper import log_norm_nmf, norm_nmf

import numpy as np
from scipy import sparse

def run_state_estimation(data, clusters, dist='Poiss', reps=1, **kwargs):
    """
    Runs state estimation for multiple initializations, returning the result with the highest log-likelihood. All the arguments are passed to the underlying state estimation functions (poisson_estimate_state, nb_estimate_state, zip_estimate_state).

    Args:
        data (array): genes x cells
        clusters (int): number of mixture components. If this is set to 0, this is automatically estimated using gap score.
        dist (str, optional): Distribution used in state estimation. Options: 'Poiss', 'NB', 'ZIP', 'LogNorm', 'Gaussian'. Default: 'Poiss'
        reps (int, optional): number of times to run the state estimation, taking the result with the highest log-likelihood.
        **kwargs: arguments to pass to the underlying state estimation function.

    Returns:
        M (array): genes x clusters - state means
        W (array): clusters x cells - state mixing components for each cell
        ll (float): final log-likelihood
    """
    clusters = int(clusters)
    func = poisson_estimate_state
    dist = dist.lower()
    if dist=='poiss' or dist=='poisson':
        pass
    elif dist=='nb':
        func = nb_estimate_state
    elif dist=='zip':
        func = zip_estimate_state
    elif dist=='lognorm' or dist=='log-normal' or dist=='lognormal':
        func = log_norm_nmf
    elif dist=='gaussian' or dist=='norm' or dist=='normal':
        func = norm_nmf
    else:
        print('dist should be one of Poiss, NB, ZIP, LogNorm, or Gaussian. Using Poiss.')
    # TODO: estimate number of clusters
    if clusters == 0:
        from .gap_score import run_gap_k_selection, preproc_data
        data_tsvd = preproc_data(data, gene_subset=False)
        max_k, gap_vals, sk_vals = run_gap_k_selection(data_tsvd,
                k_min=1, k_max=50, skip=5, B=6)
        clusters = min(max_k, data.shape[0] - 1, data.shape[1] - 1)
    best_ll = np.inf
    best_M = None
    best_W = None
    for i in range(reps):
        results = func(data, clusters, **kwargs)
        M = results[0]
        W = results[1]
        if dist=='NB':
            ll = results[3]
        else:
            ll = results[2]
        if ll < best_ll:
            best_ll = ll
            best_M = M
            best_W = W
    return best_M, best_W, best_ll


