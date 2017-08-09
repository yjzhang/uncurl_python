# TODO: evaluation

from collections import Counter

import numpy as np

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
