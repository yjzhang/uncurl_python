# state estimation with poisson convex mixture model

from .state_estimation import poisson_estimate_state
from .nb_state_estimation import nb_estimate_state
from .zip_state_estimation import zip_estimate_state

import numpy as np
from scipy import sparse

def run_state_estimation(data, clusters, dist='Poiss', reps=1, init_means=None, init_weights=None, method='NoLips', max_iters=10, tol=1e-10, disp=True, inner_max_iters=100, normalize=True, initialization='cluster'):
    """
    Runs state estimation for multiple initializations, returning the result with the highest log-likelihood. All the arguments are passed to the underlying state estimation functions (poisson_estimate_state, nb_estimate_state, zip_estimate_state).

    Args:
        data (array): genes x cells
        clusters (int): number of mixture components
        dist (str, optional): Distribution used in state estimation. Options: 'Poiss', 'NB', 'ZIP'.
        reps (int, optional): number of random initializations. Default: 10.
        init_means (array, optional): initial centers - genes x clusters. Default: from Poisson kmeans
        init_weights (array, optional): initial weights - clusters x cells, or assignments as produced by clustering. Default: from Poisson kmeans
        method (str, optional): optimization method. Current options are 'NoLips' and 'L-BFGS-B'. Default: 'NoLips'. Only for Poisson; other methods always use L-BFGS-B.
        max_iters (int, optional): maximum number of iterations. Default: 10
        tol (float, optional): if both M and W change by less than tol (RMSE), then the iteration is stopped. Default: 1e-10
        disp (bool, optional): whether or not to display optimization parameters. Default: True
        inner_max_iters (int, optional): Number of iterations to run in the optimization subroutine for M and W. Default: 100
        normalize (bool, optional): True if the resulting W should sum to 1 for each cell. Default: True.
        initialization (str, optional): If initial means and weights are not provided
        , this describes how they are initialized. Options: 'cluster' (poisson kmeans), 'kmpp' (kmeans++ for means, random weights). Default: cluster. Currently only for Poisson state estimation.

    Returns:
        M (array): genes x clusters - state means
        W (array): clusters x cells - state mixing components for each cell
        ll (float): final log-likelihood
    """
    func = poisson_estimate_state
    if dist=='Poiss':
        pass
    elif dist=='NB':
        func = nb_estimate_state
    elif dist=='ZIP':
        func = zip_estimate_state
    else:
        print('dist should be one of Poiss, NB, or ZIP. Using Poiss.')
    best_ll = np.inf
    best_M = None
    best_W = None
    for i in range(reps):
        results = func(data, clusters,
                init_means=init_means, init_weights=init_weights,
                method=method, max_iters=max_iters, tol=tol, disp=disp,
                inner_max_iters=inner_max_iters, normalize=normalize,
                initialization=initialization)
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


