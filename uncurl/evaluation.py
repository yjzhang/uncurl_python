from collections import Counter

import numpy as np
from sklearn.neighbors import BallTree

def purity(labels, true_labels):
    """
    Calculates the purity score for the given labels.

    Args:
        labels (array): 1D array of integers
        true_labels (array): 1D array of integers - true labels

    Returns:
        purity score - a float bewteen 0 and 1. Closer to 1 is better.
    """
    purity = 0.0
    for i in set(labels):
        indices = (labels==i)
        true_clusters = true_labels[indices]
        if len(true_clusters)==0:
            continue
        counts = Counter(true_clusters)
        lab, count = counts.most_common()[0]
        purity += count
    return float(purity)/len(labels)

def nne(dim_red, true_labels):
    """
    Calculates the nearest neighbor accuracy (basically leave-one-out cross
    validation with a 1NN classifier).

    Args:
        dim_red (array): dimensions (k, cells)
        true_labels (array): 1d array of integers

    Returns:
        Nearest neighbor accuracy - fraction of points for which the 1NN
        1NN classifier returns the correct value.
    """
    # use sklearn's BallTree
    bt = BallTree(dim_red.T)
    correct = 0
    for i, l in enumerate(true_labels):
        dist, ind = bt.query([dim_red[:,i]], k=2)
        closest_cell = ind[0, 1]
        if true_labels[closest_cell] == l:
            correct += 1
    return float(correct)/len(true_labels)

def mdl(ll, k, data):
    """
    Returns the minimum description length score of the model given its
    log-likelihood and k, the number of cell types.

    a lower cost is better...
    """

    """
    N - no. of genes
    n - no. of cells 
    k - no. of cell types
    R - sum(Dataset) i.e. total no. of reads

    function TotCost = TotBits(N,m,p,R,C)
        # C is the cost from the cost function
        TotCost = C + (N*m + m*p)*(log(R/(N*p)));
    """
    N, m = data.shape
    cost = ll + (N*m + m*k)*(np.log(data.sum()/(N*k)))
    return cost
