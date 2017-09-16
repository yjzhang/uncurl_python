# general framework for running purity experiments on 10x dataset

# two steps: dimensionality reduction/preprocessing, and clustering
# preprocessing -> dim_red -> clustering?
# ex. uncurl_mw -> tsne -> km???

# preprocessing returns a matrix of shape (k, cells), where k <= genes

import numpy as np

from state_estimation import poisson_estimate_state
from evaluation import purity
from preprocessing import cell_normalize
from ensemble import nmf_ensemble, nmf_kfold, nmf_tsne, poisson_se_tsne
from clustering import poisson_cluster

from scipy import sparse
from scipy.special import log1p

from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

import SIMLR


class Preprocess(object):
    """
    Pre-processing methods take in a genes x cells data matrix of integer
    counts, and return a k x cells matrix, where k <= genes.

    Preprocessing methods can return multiple outputs. the outputs are

    If k=2, then the method can be used for visualization...

    This class represents a 'blank' preprocessing.
    """

    def __init__(self, **params):
        if 'output_names' in params:
            self.output_names = params['output_names']
            params.pop('output_names')
        self.params = params
        if not hasattr(self, 'output_names'):
            self.output_names = ['Data']

    def run(self, data):
        """
        should return a list of output matrices of the same length
        as self.output_names, and an objective value.

        data is of shape (genes, cells).
        """
        return [data], 0

class Log(Preprocess):
    """
    Takes the natural log of the data+1.
    """

    def __init__(self, **params):
        self.output_names = ['LogData']
        super(Log, self).__init__(**params)

    def run(self, data):
        if sparse.issparse(data):
            return [data.log1p()], 0
        else:
            return [np.log1p(data)], 0

class LogNorm(Preprocess):
    """
    First, normalizes the counts per cell, and then takes log(normalized_counts+1).
    """

    def __init__(self, **params):
        self.output_names = ['LogNorm']
        super(LogNorm, self).__init__(**params)

    def run(self, data):
        data_norm = cell_normalize(data)
        if sparse.issparse(data_norm):
            return [data_norm.log1p()], 0
        else:
            return [np.log1p(data_norm)], 0

class TSVD(Preprocess):
    """
    Runs truncated SVD on the data. the input param k is the number of
    dimensions.
    """

    def __init__(self, **params):
        self.output_names = ['TSVD']
        self.tsvd = TruncatedSVD(params['k'])
        super(TSVD, self).__init__(**params)

    def run(self, data):
        return [self.tsvd.fit_transform(data.T)], 0


class PoissonSE(Preprocess):
    """
    Runs Poisson State Estimation, returning W and MW.

    Requires a 'k' parameter.
    """

    def __init__(self, **params):
        self.output_names = ['Poisson_W', 'Poisson_MW']
        super(PoissonSE, self).__init__(**params)

    def run(self, data):
        M, W, ll = poisson_estimate_state(data, **self.params)
        return [W, M.dot(W)], ll

class LogNMF(Preprocess):
    """
    Runs NMF on log(normalize(data)+1), returning H and W*H.

    Requires a 'k' parameter, which is the rank of the matrices.
    """

    def __init__(self, **params):
        self.output_names = ['logNMF_H', 'logNMF_WH']
        super(LogNMF, self).__init__(**params)
        self.nmf = NMF(params['k'])

    def run(self, data):
        data_norm = cell_normalize(data)
        if sparse.issparse(data_norm):
            data_norm = data_norm.log1p()
        else:
            data_norm = log1p(data_norm)
        W = self.nmf.fit_transform(data_norm)
        H = self.nmf.components_
        if sparse.issparse(data_norm):
            ws = sparse.csr_matrix(W)
            hs = sparse.csr_matrix(H)
            cost = 0.5*((data_norm - ws.dot(hs)).power(2)).sum()
        else:
            cost = 0.5*((data_norm - W.dot(H))**2).sum()
        return [H, W.dot(H)], cost

class BasicNMF(Preprocess):
    """
    Runs NMF on data, returning H and W*H.

    Requires a 'k' parameter, which is the rank of the matrices.
    """

    def __init__(self, **params):
        self.output_names = ['NMF_H', 'NMF_WH']
        super(BasicNMF, self).__init__(**params)
        self.nmf = NMF(params['k'])

    def run(self, data):
        W = self.nmf.fit_transform(data)
        H = self.nmf.components_
        if sparse.issparse(data):
            ws = sparse.csr_matrix(W)
            hs = sparse.csr_matrix(H)
            cost = 0.5*((data - ws.dot(hs)).power(2)).sum()
        else:
            cost = 0.5*((data - W.dot(H))**2).sum()
        return [H, W.dot(H)], cost

class EnsembleNMF(Preprocess):
    """
    Runs Ensemble NMF on log(data+1), returning the consensus
    results for H and W*H.

    Requires a 'k' parameter, which is the rank of the matrices.
    """

    def __init__(self, **params):
        self.output_names = ['H', 'WH']
        super(EnsembleNMF, self).__init__(**params)

    def run(self, data):
        data_norm = cell_normalize(data)
        if sparse.issparse(data_norm):
            data_norm = data_norm.log1p()
        else:
            data_norm = log1p(data_norm)
        W, H = nmf_ensemble(data_norm, **self.params)
        if sparse.issparse(data_norm):
            ws = sparse.csr_matrix(W)
            hs = sparse.csr_matrix(H)
            cost = 0.5*((data_norm - ws.dot(hs)).power(2)).sum()
        else:
            cost = 0.5*((data_norm - W.dot(H))**2).sum()
        return [H, W.dot(H)], cost

class KFoldNMF(Preprocess):
    """
    Runs K-fold ensemble NMF on log(data+1), returning the consensus
    results for H and W*H.

    Requires a 'k' parameter, which is the rank of the matrices.
    """

    def __init__(self, **params):
        self.output_names = ['H', 'WH']
        super(KFoldNMF, self).__init__(**params)

    def run(self, data):
        data_norm = cell_normalize(data)
        if sparse.issparse(data_norm):
            data_norm = data_norm.log1p()
        else:
            data_norm = log1p(data_norm)
        W, H = nmf_kfold(data_norm, **self.params)
        if sparse.issparse(data_norm):
            ws = sparse.csr_matrix(W)
            hs = sparse.csr_matrix(H)
            cost = 0.5*((data_norm - ws.dot(hs)).power(2)).sum()
        else:
            cost = 0.5*((data_norm - W.dot(H))**2).sum()
        return [H, W.dot(H)], cost

class EnsembleTsneNMF(Preprocess):
    """
    Runs tsne-based ensemble NMF
    """

    def __init__(self, **params):
        self.output_names = ['Ensemble_NMF_H', 'Ensemble_NMF_WH']
        super(EnsembleTsneNMF, self).__init__(**params)

    def run(self, data):
        data_norm = cell_normalize(data)
        if sparse.issparse(data_norm):
            data_norm = data_norm.log1p()
        else:
            data_norm = log1p(data_norm)
        W, H = nmf_tsne(data_norm, **self.params)
        if sparse.issparse(data_norm):
            ws = sparse.csr_matrix(W)
            hs = sparse.csr_matrix(H)
            cost = 0.5*((data_norm - ws.dot(hs)).power(2)).sum()
        else:
            cost = 0.5*((data_norm - W.dot(H))**2).sum()
        return [H, W.dot(H)], cost

class EnsembleTsnePoissonSE(Preprocess):
    """
    Runs tsne-based ensemble Poisson state estimation
    """

    def __init__(self, **params):
        self.output_names = ['Ensemble_W', 'Ensemble_MW']
        super(EnsembleTsnePoissonSE, self).__init__(**params)

    def run(self, data):
        M, W, obj = poisson_se_tsne(data, **self.params)
        return [W, M.dot(W)], obj

class Simlr(Preprocess):

    def __init__(self, **params):
        self.output_names = ['PCA50_SIMLR']
        # TODO: make params tunable... what do these numbers even mean???
        self.simlr = SIMLR.SIMLR_LARGE(8, 30, 0)
        super(Simlr, self).__init__(**params)

    def run(self, data):
        X = SIMLR.helper.fast_pca(data.T, 50)
        S, F, val, ind = self.simlr.fit(X)
        return [F.T], 0

class Magic(Preprocess):
    # TODO: this requires python 3

    def __init__(self, **params):
        pass

    def run(self, data):
        pass

class Cluster(object):
    """
    Clustering methods take in a matrix of shape k x cells, and
    return an array of integers in (0, n_classes-1).

    They should be able to run on the output of pre-processing...
    """

    def __init__(self, n_classes, **params):
        self.n_classes = n_classes
        self.params = params
        self.name = ''

    def run(self, data):
        pass

class Argmax(Cluster):

    def __init__(self, n_classes, **params):
        super(Argmax, self).__init__(n_classes, **params)
        self.name = 'argmax'

    def run(self, data):
        assert(data.shape[0]==self.n_classes)
        return data.argmax(0)

class KM(Cluster):
    """
    k-means clustering
    """

    def __init__(self, n_classes, **params):
        super(KM, self).__init__(n_classes, **params)
        self.name = 'km'
        self.km = KMeans(n_classes)

    def run(self, data):
        return self.km.fit_predict(data.T)

class PoissonCluster(Cluster):
    """
    Poisson k-means clustering
    """

    def __init__(self, n_classes, **params):
        super(PoissonCluster, self).__init__(n_classes, **params)
        self.name = 'poisson_km'

    def run(self, data):
        assignments, means = poisson_cluster(data, self.n_classes)
        return assignments


class PcaKm(Cluster):
    """
    PCA + kmeans

    Requires a parameter k, where k is the dimensionality
    of PCA.
    """

    def __init__(self, n_classes, **params):
        super(PcaKm, self).__init__(n_classes, **params)
        self.pca = PCA(params['k'])
        self.km = KMeans(n_classes)
        self.name = 'pca_km'

    def run(self, data):
        data_pca = self.pca.fit_transform(data.T)
        labels = self.km.fit_predict(data_pca)
        return labels

class TsneKm(Cluster):
    """
    TSNE(2) + Kmeans
    """

    def __init__(self, n_classes, **params):
        super(TsneKm, self).__init__(n_classes, **params)
        self.tsne = TSNE(2)
        self.km = KMeans(n_classes)
        self.name = 'tsne_km'

    def run(self, data):
        if sparse.issparse(data):
            data = data.toarray()
        data_tsne = self.tsne.fit_transform(data.T)
        labels = self.km.fit_predict(data_tsne)
        return labels

class SimlrKm(Cluster):
    """
    Fast minibatch Kmeans from the simlr library
    """

    def __init__(self, n_classes, **params):
        super(SimlrKm, self).__init__(n_classes, **params)
        self.simlr = SIMLR.SIMLR_LARGE(8, 30, 0)
        self.name = 'km'

    def run(self, data):
        y_pred = self.simlr.fast_minibatch_kmeans(data.T, 8)
        return y_pred

def run_experiment(methods, data, n_classes, true_labels, n_runs=10):
    """
    runs a pre-processing + clustering experiment...

    Args:
        methods: list of pairs of Preprocess, (list of) Cluster objects
        data: genes x cells array

    Returns:
        purities
        names
    """
    results = []
    names = []
    for i in range(n_runs):
        print('run {0}'.format(i))
        purities = []
        r = 0
        for preproc, cluster in methods:
            preprocessed, ll = preproc.run(data)
            for name, pre in zip(preproc.output_names, preprocessed):
                if isinstance(cluster, Cluster):
                    try:
                        labels = cluster.run(pre)
                        purities.append(purity(labels, true_labels))
                        if i==0:
                            names.append(name + '_' + cluster.name)
                        print(names[r])
                        print(purity(labels, true_labels))
                        r += 1
                    except:
                        print('failed to do clustering')
                elif type(cluster) == list:
                    for c in cluster:
                        try:
                            labels = c.run(pre)
                            purities.append(purity(labels, true_labels))
                            if i==0:
                                names.append(name + '_' + c.name)
                            print(names[r])
                            print(purity(labels, true_labels))
                            r += 1
                        except:
                            print('failed to do clustering')
        print('\t'.join(names))
        print('purities: ' + '\t'.join(map(str, purities)))
        results.append(purities)
    return results, names
