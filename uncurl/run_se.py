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
        clusters (int): number of mixture components
        dist (str, optional): Distribution used in state estimation. Options: 'Poiss', 'NB', 'ZIP', 'LogNorm', 'Gaussian'. Default: 'Poiss'
        reps (int, optional): number of times to run the state estimation, taking the result with the highest log-likelihood.
        **kwargs: arguments to pass to the underlying state estimation function.

    Returns:
        M (array): genes x clusters - state means
        W (array): clusters x cells - state mixing components for each cell
        ll (float): final log-likelihood
    """
    func = poisson_estimate_state
    dist = dist.lower()
    if dist=='poiss' or dist=='poisson':
        pass
    elif dist=='nb':
        func = nb_estimate_state
    elif dist=='zip':
        func = zip_estimate_state
    elif dist=='lognorm':
        func = log_norm_nmf
    elif dist=='gaussian' or dist=='norm':
        func = norm_nmf
    else:
        print('dist should be one of Poiss, NB, ZIP, LogNorm, or Gaussian. Using Poiss.')
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


