"""
Using gap score to determine optimal cluster number
"""
import numpy as np
from sklearn.cluster import KMeans

def preproc_data(data, gene_subset=False, **kwargs):
    """
    basic data preprocessing before running gap score

    Assumes that data is a matrix of shape (genes, cells).

    Returns a matrix of shape (cells, 8), using the first 8 SVD
    components. Why 8? It's an arbitrary selection...
    """
    import uncurl
    from uncurl.preprocessing import log1p, cell_normalize
    from sklearn.decomposition import TruncatedSVD
    data_subset = data
    if gene_subset:
        gene_subset = uncurl.max_variance_genes(data)
        data_subset = data[gene_subset, :]
    tsvd = TruncatedSVD(min(8, data_subset.shape[0] - 1))
    data_tsvd = tsvd.fit_transform(log1p(cell_normalize(data_subset)).T)
    return data_tsvd

def calculate_bounding_box(data):
    """
    Returns a 2 x m array indicating the min and max along each
    dimension.
    """
    mins = data.min(0)
    maxes = data.max(0)
    return mins, maxes

def calculate_gap(data, clustering, km, B=50, **kwargs):
    """
    See: https://datasciencelab.wordpress.com/2013/12/27/finding-the-k-in-k-means-clustering/

    https://web.stanford.edu/~hastie/Papers/gap.pdf

    Returns two results: the gap score, and s_k.
    """
    k = len(set(clustering))
    Wk = km.inertia_
    mins, maxes = calculate_bounding_box(data)
    Wk_est = []
    for i in range(B):
        data_sample = (maxes-mins)*np.random.random(data.shape) + mins
        km_b = KMeans(k)
        km_b.fit_predict(data_sample)
        Wk_est.append(km_b.inertia_)
    Wk_est = np.log(np.array(Wk_est))
    Wk_mean = np.mean(Wk_est)
    Wk_std = np.std(Wk_est)
    gap = Wk_mean - np.log(Wk)
    sk = np.sqrt(1 + 1.0/B)*Wk_std
    return gap, sk


def run_gap_k_selection(data, k_min=1, k_max=50, B=5,
        skip=5, **kwargs):
    """
    Runs gap score for all k from k_min to k_max.
    """
    if k_min == k_max:
        return k_min
    gap_vals = []
    sk_vals = []
    k_range = list(range(k_min, k_max, skip))
    min_k = 0
    min_i = 0
    for i, k in enumerate(k_range):
        km = KMeans(k)
        clusters = km.fit_predict(data)
        gap, sk = calculate_gap(data, clusters, km, B=B)
        if len(gap_vals) > 1:
            if gap_vals[-1] >= gap - (skip+1)*sk:
                min_i = i
                min_k = k_range[i-1]
                break
                #return k_range[-1], gap_vals, sk_vals
        gap_vals.append(gap)
        sk_vals.append(sk)
    if min_k == 0:
        min_k = k_max
    if skip == 1:
        return min_k, gap_vals, sk_vals
    gap_vals = []
    sk_vals = []
    for k in range(min_k - skip, min_k + skip):
        km = KMeans(k)
        clusters = km.fit_predict(data)
        gap, sk = calculate_gap(data, clusters, km, B=B)
        if len(gap_vals) > 1:
            if gap_vals[-1] >= gap - sk:
                min_k = k-1
                return min_k, gap_vals, sk_vals
        gap_vals.append(gap)
        sk_vals.append(sk)
    return k, gap_vals, sk_vals

