# general framework for running purity experiments on 10x dataset

# two steps: dimensionality reduction/preprocessing, and clustering
# preprocessing -> dim_red -> clustering?
# ex. uncurl_mw -> tsne -> km???

# preprocessing returns a matrix of shape (k, cells), where k <= genes

import numpy as np

from state_estimation import poisson_estimate_state
from evaluation import purity
from preprocessing import cell_normalize
from ensemble import nmf_ensemble, nmf_kfold

from scipy import sparse
from scipy.special import log1p

from sklearn.decomposition import NMF
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
        self.output_names = []
        self.params = params

    def run(self, data):
        """
        should return a list of output matrices of the same length
        as self.output_names.
        """
        return data

class PoissonSE(Preprocess):
    """
    Runs Poisson State Estimation, returning W and MW.

    Requires a 'k' parameter.
    """

    def __init__(self, **params):
        self.output_names = ['Poisson_W', 'Poisson_MW']
        self.params = params

    def run(self, data):
        W, M, ll = poisson_estimate_state(data, **self.params)
        return [W, M.dot(W)]

class LogNMF(Preprocess):
    """
    Runs NMF on log(data+1), returning H and W*H.

    Requires a 'k' parameter, which is the rank of the matrices.
    """

    def __init__(self, **params):
        self.output_names = ['logNMF_H', 'logNMF_WH']
        self.params = params
        self.nmf = NMF(params['k'])

    def run(self, data):
        data_norm = cell_normalize(data)
        if sparse.issparse(data_norm):
            data_norm = data_norm.log1p()
        else:
            data_norm = log1p(data_norm)
        W = self.nmf.fit_transform(data_norm)
        H = self.nmf.components_
        return [H, W.dot(H)]

class EnsembleNMF(Preprocess):
    """
    Runs Ensemble NMF on log(data+1), returning the consensus
    results for H and W*H.

    Requires a 'k' parameter, which is the rank of the matrices.
    """

    def __init__(self, **params):
        self.output_names = ['H', 'WH']
        self.params = params

    def run(self, data):
        data_norm = cell_normalize(data)
        if sparse.issparse(data_norm):
            data_norm = data_norm.log1p()
        else:
            data_norm = log1p(data_norm)
        W, H = nmf_ensemble(data_norm, **self.params)
        return [H, W.dot(H)]

class KFoldNMF(Preprocess):
    """
    Runs K-fold ensemble NMF on log(data+1), returning the consensus
    results for H and W*H.

    Requires a 'k' parameter, which is the rank of the matrices.
    """

    def __init__(self, **params):
        self.output_names = ['H', 'WH']
        self.params = params

    def run(self, data):
        data_norm = cell_normalize(data)
        if sparse.issparse(data_norm):
            data_norm = data_norm.log1p()
        else:
            data_norm = log1p(data_norm)
        W, H = nmf_kfold(data_norm, **self.params)
        return [H, W.dot(H)]

class EnsembleTsneNMF(Preprocess):
    """
    """

    def __init__(self, **params):
        self.output_names = ['Ensemble_H', 'Ensemble_WH']
        self.params = params

    def run(self, data):
        data_norm = cell_normalize(data)
        if sparse.issparse(data_norm):
            data_norm = data_norm.log1p()
        else:
            data_norm = log1p(data_norm)


class Simlr(Preprocess):

    def __init__(self, **params):
        self.output_names = ['PCA50_SIMLR']
        self.params = params
        # TODO: make params tunable... what do these numbers even mean???
        self.simlr = SIMLR.SIMLR_LARGE(8, 30, 0)

    def run(self, data):
        X = SIMLR.helper.fast_pca(data.T, 50)
        S, F, val, ind = self.simlr.fit(X)
        return [F.T]

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
        for preproc, cluster in methods:
            preprocessed = preproc.run(data)
            for name, pre in zip(preproc.output_names, preprocessed):
                if isinstance(cluster, Cluster):
                    try:
                        labels = cluster.run(pre)
                        purities.append(purity(labels, true_labels))
                        if i==0:
                            names.append(name + '_' + cluster.name)
                    except:
                        pass
                elif type(cluster) == list:
                    for c in cluster:
                        try:
                            labels = c.run(pre)
                            purities.append(purity(labels, true_labels))
                            if i==0:
                                names.append(name + '_' + c.name)
                        except:
                            pass
        print('\t'.join(map(str, purities)))
        results.append(purities)
    return results, names
