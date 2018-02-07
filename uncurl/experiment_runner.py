# general framework for running purity experiments on 10x dataset

# two steps: dimensionality reduction/preprocessing, and clustering
# preprocessing -> dim_red -> clustering?
# ex. uncurl_mw -> tsne -> km???

# preprocessing returns a matrix of shape (k, cells), where k <= genes
from __future__ import print_function

import time
import sys

try:
    # optional dependencies...
    import matplotlib.pyplot as plt
except:
    pass

import numpy as np
from scipy import sparse
from scipy.special import log1p

from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.metrics.cluster import adjusted_rand_score as ari

from sklearn.decomposition import NMF, TruncatedSVD, PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.cluster.bicluster import SpectralBiclustering, SpectralCoclustering

try:
    import Cluster_Ensembles as CE
except:
    pass

# optional dependencies?
try:
    import SIMLR
    from ZIFA import ZIFA
    from ZIFA import block_ZIFA
except:
    # optional dependencies?
    pass

# magic (requires python 3)
try:
    import pandas as pd
    import magic
except:
    pass

from .state_estimation import poisson_estimate_state
from .dimensionality_reduction import dim_reduce
from .evaluation import purity, nne
from .preprocessing import cell_normalize
try:
    from . import ensemble
    from .ensemble import nmf_ensemble, nmf_kfold, nmf_tsne, poisson_se_tsne, poisson_se_multiclust, lightlda_se_tsne
except:
    print('unable to import ensemble methods.')
from .clustering import poisson_cluster
from .lightlda_utils import lightlda_estimate_state
from .plda_utils import plda_estimate_state
from .vis import visualize_dim_red

from uncurl.sparse_utils import symmetric_kld, jensen_shannon


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
        return [self.tsvd.fit_transform(data.T).T], 0

class Tsne(Preprocess):
    """
    2d tsne dimensionality reduction - tsne always uses 2d

    metric is a string that could be any metric usable with tsne, or 'kld'
    or 'jensen-shannon'
    """

    def __init__(self, metric='euclidean', **params):
        """
        metric (str) can be any metric usable with tsne.
        """
        self.output_names = ['TSNE']
        if metric != 'euclidean':
            self.output_names = ['TSNE_' + metric]
        if metric=='kld':
            metric = symmetric_kld
        elif metric == 'jensen-shannon':
            metric = jensen_shannon
        self.tsne = TSNE(2, metric=metric)
        super(Tsne, self).__init__(**params)

    def run(self, data):
        if sparse.issparse(data):
            data = data.toarray()
        return [self.tsne.fit_transform(data.T).T], 0

class Pca(Preprocess):
    """
    PCA preprocessing
    """

    def __init__(self, **params):
        self.output_names = ['PCA']
        self.pca = PCA(params['k'])
        super(Pca, self).__init__(**params)

    def run(self, data):
        if sparse.issparse(data):
            data = data.toarray()
        return [self.pca.fit_transform(data.T).T], 0

class Zifa(Preprocess):
    """
    ZIFA preprocessing
    """

    def __init__(self, **params):
        self.output_names = ['ZIFA']
        self.k = params['k']
        super(Zifa, self).__init__(**params)

    def run(self, data):
        if sparse.issparse(data):
            data = data.toarray()
        data = np.log1p(data)
        Z, model_params = block_ZIFA.fitModel(data.T, self.k)
        return [Z.T], 0


class PoissonSE(Preprocess):
    """
    Runs Poisson State Estimation, returning W and MW.

    Requires a 'k' parameter.

    Optional args: return_m=True: returns M in outputs
    return_mw=True: returns MW in outputs
    """

    def __init__(self, return_w=True, return_m=False, return_mw=False,
            return_mds=False, normalize_data=False, **params):
        self.output_names = []
        self.return_w = return_w
        if return_w:
            self.output_names.append('Poisson_W')
        self.return_m = return_m
        if return_m:
            self.output_names.append('Poisson_M')
        self.return_mw = return_mw
        if return_mw:
            self.output_names.append('Poisson_MW')
        self.return_mds = return_mds
        if return_mds:
            self.output_names.append('Poisson_MDS')
        self.normalize_data = normalize_data
        super(PoissonSE, self).__init__(**params)

    def run(self, data):
        """
        Returns:
            list of W, M*W
            ll
        """
        if self.normalize_data:
            data = cell_normalize(data)
        M, W, ll = poisson_estimate_state(data, **self.params)
        outputs = []
        if self.return_w:
            outputs.append(W)
        if self.return_m:
            outputs.append(M)
        if self.return_mw:
            outputs.append(M.dot(W))
        if self.return_mds:
            X = dim_reduce(M, W, 2)
            outputs.append(X.T.dot(W))
        return outputs, ll


class LightLDASE(Preprocess):
    """
    Runs LightLDA State Estimation, returning W and MW.
    Requires a 'k' parameter.
    """

    def __init__(self, **params):
        self.output_names = ['LightLDA_W']
        self.return_w = True
        self.return_m = False
        self.return_mw = False
        self.return_mds = False
        if 'return_mw' in params and params['return_mw']:
            self.output_names.append('LightLDA_MW')
            self.return_mw = True
            params.pop('return_mw')
        if 'return_mds' in params and params['return_mds']:
            self.output_names.append('LightLDA_MDS')
            self.return_mds = True
            params.pop('return_mds')
        super(LightLDASE, self).__init__(**params)

    def run(self, data):
        M, W, ll = lightlda_estimate_state(data, **self.params)
        output = [W]
        if self.return_mw:
            output.append(M.dot(W))
        if self.return_mds:
            X = dim_reduce(M, W, 2)
            output.append(X.T.dot(W))
        return output, ll

class PLDASE(Preprocess):
    """
    Runs PLDA State Estimation, returning W and MW.
    Requires a 'k' parameter.
    """

    def __init__(self, **params):
        self.output_names = ['PLDA_W']
        if 'return_mw' in params and params['return_mw']:
            self.output_names.append('PLDA_MW')
            self.return_mw = True
            params.pop('return_mw')
        if 'return_mds' in params and params['return_mds']:
            self.output_names.append('PLDA_MDS')
            self.return_mds = True
            params.pop('return_mds')
        super(PLDASE, self).__init__(**params)

    def run(self, data):
        M, W = plda_estimate_state(data, **self.params)
        output = [W]
        if self.return_mw:
            output.append(M.dot(W))
        if self.return_mds:
            X = dim_reduce(M, W, 2)
            output.append(X.T.dot(W))
        return output, 0



class EnsembleTsneLightLDASE(Preprocess):
    """
    Runs tsne-based LightLDA Poisson state estimation
    """

    def __init__(self, **params):
        self.output_names = ['LightLDA_Ensemble_W']
        if 'return_mw' in params and params['return_mw']:
            self.output_names.append('LightLDA_Ensemble_MW')
            self.return_mw = True
            params.pop('return_mw')
        if 'return_mds' in params and params['return_mds']:
            self.output_names.append('LightLDA_Ensemble_MDS')
            self.return_mds = True
            params.pop('return_mds')
        super(EnsembleTsneLightLDASE, self).__init__(**params)

    def run(self, data):
        M, W, ll = lightlda_se_tsne(data, **self.params)
        output = [W]
        if self.return_mw:
            output.append(M.dot(W))
        if self.return_mds:
            X = dim_reduce(M, W, 2)
            output.append(X.T.dot(W))
        return output, ll


class LogNMF(Preprocess):
    """
    Runs NMF on log(normalize(data)+1), returning H and W*H.

    Requires a 'k' parameter, which is the rank of the matrices.
    """

    def __init__(self, return_h=True, return_w=False, return_mds=False, return_wh=False, **params):
        super(LogNMF, self).__init__(**params)
        self.output_names = []
        self.nmf = NMF(params['k'])
        self.return_h = return_h
        if return_h:
            self.output_names.append('logNMF_H')
        self.return_w = return_w
        if return_w:
            self.output_names.append('logNMF_W')
        self.return_mds = return_mds
        if return_mds:
            self.output_names.append('logNMF_MDS')
        self.return_wh = return_wh
        if return_wh:
            self.output_names.append('logNMF_WH')

    def run(self, data):
        data_norm = cell_normalize(data)
        if sparse.issparse(data_norm):
            data_norm = data_norm.log1p()
        else:
            data_norm = log1p(data_norm)
        W = self.nmf.fit_transform(data_norm)
        H = self.nmf.components_
        if sparse.issparse(data_norm):
            cost = 0
            #ws = sparse.csr_matrix(W)
            #hs = sparse.csr_matrix(H)
            #cost = 0.5*((data_norm - ws.dot(hs)).power(2)).sum()
        else:
            cost = 0.5*((data_norm - W.dot(H))**2).sum()
        if 'normalize_h' in self.params:
            print('normalize h')
            H = H/H.sum(0)
        output = []
        if self.return_h:
            output.append(H)
        if self.return_w:
            output.append(W)
        if self.return_wh:
            output.append(W.dot(H))
        if self.return_mds:
            X = dim_reduce(W, H, 2)
            output.append(X.T.dot(H))
        return output, cost

class BasicNMF(Preprocess):
    """
    Runs NMF on data, returning H and W*H.

    Requires a 'k' parameter, which is the rank of the matrices.
    """

    def __init__(self, return_h=True, return_w=False, return_mds=False, return_wh=False, **params):
        super(BasicNMF, self).__init__(**params)
        self.nmf = NMF(params['k'])
        self.output_names = []
        self.return_h = return_h
        if return_h:
            self.output_names.append('NMF_H')
        self.return_w = return_w
        if return_w:
            self.output_names.append('NMF_W')
        self.return_mds = return_mds
        if return_mds:
            self.output_names.append('NMF_MDS')
        self.return_wh = return_wh
        if return_wh:
            self.output_names.append('NMF_WH')

    def run(self, data):
        data_norm = cell_normalize(data)
        W = self.nmf.fit_transform(data_norm)
        H = self.nmf.components_
        if sparse.issparse(data):
            ws = sparse.csr_matrix(W)
            hs = sparse.csr_matrix(H)
            cost = 0.5*((data - ws.dot(hs)).power(2)).sum()
        else:
            cost = 0.5*((data - W.dot(H))**2).sum()
        if 'normalize_h' in self.params:
            H = H/H.sum(0)
        output = []
        if self.return_h:
            output.append(H)
        if self.return_w:
            output.append(W)
        if self.return_wh:
            output.append(W.dot(H))
        if self.return_mds:
            X = dim_reduce(W, H, 2)
            output.append(X.T.dot(H))
        return output, cost

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
        outputs = []
        outputs.append(W)
        if 'return_m' in self.params and self.params['return_m']:
            outputs.append(M)
        if 'return_mw' in self.params and self.params['return_mw']:
            outputs.append(M.dot(W))
        return outputs, obj

class EnsembleTSVDPoissonSE(Preprocess):
    """
    Runs Poisson state estimation initialized from 8 runs of tsvd-km.

    params: k - dimensionality
    """

    def __init__(self, **params):
        self.output_names = ['TSVDEnsemble_W']
        self.return_m = False
        self.return_mw = False
        self.return_mds = False
        if 'return_m' in params and params['return_m']:
            self.output_names.append('TSVDEnsemble_M')
            self.return_m = True
            params.pop('return_m')
        if 'return_mw' in params and params['return_mw']:
            self.output_names.append('TSVDEnsemble_MW')
            self.return_mw = True
            params.pop('return_mw')
        if 'return_mds' in params and params['return_mds']:
            self.output_names.append('TSVDEnsemble_MDS')
            self.return_mds = True
            params.pop('return_mds')
        super(EnsembleTSVDPoissonSE, self).__init__(**params)

    def run(self, data):
        M, W, obj = poisson_se_multiclust(data, **self.params)
        outputs = []
        outputs.append(W)
        if self.return_m:
            outputs.append(M)
        if self.return_mw:
            outputs.append(M.dot(W))
        if self.return_mds:
            X = dim_reduce(M, W, 2)
            outputs.append(X.T.dot(W))
        return outputs, obj

class EnsembleClusterPoissonSE(Preprocess):
    """
    Runs Poisson state estimation initialized from the consensus
    of 10 runs of Poisson KM.

    params: k - dimensionality
    """

    def __init__(self, **params):
        self.output_names = ['ConsensusPoisson_W']
        self.return_m = False
        self.return_mw = False
        self.return_mds = False
        if 'return_m' in params and params['return_m']:
            self.output_names.append('ConsensusPoisson_M')
            self.return_m = True
            params.pop('return_m')
        if 'return_mw' in params and params['return_mw']:
            self.output_names.append('ConsensusPoisson_MW')
            self.return_mw = True
            params.pop('return_mw')
        if 'return_mds' in params and params['return_mds']:
            self.output_names.append('ConsensusPoisson_MDS')
            self.return_mds = True
            params.pop('return_mds')
        super(EnsembleClusterPoissonSE, self).__init__(**params)

    def run(self, data):
        # make the data sparse for runtime improvements in poisson clustering
        data = sparse.csc_matrix(data)
        M, W, obj = ensemble.poisson_consensus_se(data, **self.params)
        outputs = []
        outputs.append(W)
        if self.return_m:
            outputs.append(M)
        if self.return_mw:
            outputs.append(M.dot(W))
        if self.return_mds:
            X = dim_reduce(M, W, 2)
            outputs.append(X.T.dot(W))
        return outputs, obj

class Simlr(Preprocess):

    def __init__(self, **params):
        self.output_names = ['PCA50_SIMLR']
        # TODO: make params tunable... what do these numbers even mean???
        self.simlr = SIMLR.SIMLR_LARGE(params['k'], 30, 0)
        super(Simlr, self).__init__(**params)

    def run(self, data):
        X_log = np.log1p(data)
        if 'n_pca_components' in self.params:
            n_components = self.params['n_pca_components']
        else:
            n_components = 50
        X = SIMLR.helper.fast_pca(X_log.T, n_components)
        S, F, val, ind = self.simlr.fit(X)
        return [F.T], 0

class SimlrSmall(Preprocess):
    """
    Simlr for small-scale datasets (no PCA preprocessing)
    """

    def __init__(self, **params):
        self.output_names = ['SIMLR']
        # TODO: make params tunable... what do these numbers even mean???
        self.simlr = SIMLR.SIMLR(params['k'], 30, 0)
        super(Simlr, self).__init__(**params)

    def run(self, data):
        X = np.log1p(data)
        # TODO: the python implementation of simlr only contains the
        # large-scale (PCA-dependent) methods.
        if 'n_pca_components' in self.params:
            n_components = self.params['n_pca_components']
        else:
            n_components = 50
        X = SIMLR.helper.fast_pca(data.T, n_components)
        S, F, val, ind = self.simlr.fit(X)
        return [F.T], 0

class Magic(Preprocess):
    # TODO: this requires python 3

    def __init__(self, use_magic=True, use_tsne=False, use_pca=False, **params):
        self.output_names = []
        self.use_magic = use_magic
        self.use_tsne = use_tsne
        self.use_pca = use_pca
        if self.use_magic:
            self.output_names.append('magic')
        if self.use_tsne:
            self.output_names.append('magic_tsne')
        if self.use_pca:
            self.output_names.append('magic_pca')
        super(Magic, self).__init__(**params)

    def run(self, data):
        if 'n_pca_components' in self.params:
            n_components = self.params['n_pca_components']
        else:
            n_components = 20
        if sparse.issparse(data):
            data = data.toarray()
        data_array = pd.DataFrame(data.T)
        data_array.columns = data_array.columns.astype(str)
        scdata = magic.mg.SCData(pd.DataFrame(data_array), data_type='sc-seq')
        scdata = scdata.normalize_scseq_data()
        scdata.run_magic(n_pca_components=n_components, random_pca=True,
                t=6, k=30, ka=10, epsilon=1, rescale_percent=99)
        #scdata.run_tsne()
        outputs = []
        if self.use_magic:
            outputs.append(scdata.magic.data.as_matrix().T)
        if self.use_tsne:
            scdata.magic.run_tsne()
            outputs.append(scdata.magic.tsne.as_matrix().T)
        if self.use_pca:
            scdata.magic.run_pca()
            outputs.append(scdata.magic.pca.as_matrix().T)
        return outputs, 0

class LoadPreproc(Preprocess):
    """
    takes preprocessed data matrix, just return that when run is called
    """
    def __init__(self, datasets, **params):
        self.datasets = datasets
        super(LoadPreproc, self).__init__(**params)

    def run(self, data):
        return self.datasets, 0


class Cluster(object):
    """
    Clustering methods take in a matrix of shape k x cells, and
    return an array of integers in (0, n_classes-1).

    They should be able to run on the output of pre-processing...
    """

    def __init__(self, n_classes, **params):
        self.n_classes = n_classes
        self.name = ''
        if 'name' in params:
            self.name = params.pop('name')
        self.params = params

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

class DBScan(Cluster):
    """
    dbscan clustering
    """

    def __init__(self, n_classes, **params):
        super(DBScan, self).__init__(n_classes, **params)
        self.name = 'dbscan'
        self.dbscan = DBSCAN()

    def run(self, data):
        return self.dbscan.fit_predict(data.T)

class PoissonCluster(Cluster):
    """
    Poisson k-means clustering
    """

    def __init__(self, n_classes, **params):
        super(PoissonCluster, self).__init__(n_classes, **params)
        self.name = 'poisson_km'

    def run(self, data):
        assignments, means = poisson_cluster(data, self.n_classes, **self.params)
        return assignments


class PcaKm(Cluster):
    """
    PCA + kmeans

    Requires a parameter k, where k is the dimensionality
    of PCA.
    """

    def __init__(self, n_classes, use_log=False, name='pca_km', **params):
        super(PcaKm, self).__init__(n_classes, **params)
        self.use_log = use_log
        self.pca = PCA(params['k'])
        self.km = KMeans(n_classes)
        self.name = name

    def run(self, data):
        if sparse.issparse(data):
            data = data.toarray()
        if self.use_log:
            data = log1p(data)
        data_pca = self.pca.fit_transform(data.T)
        labels = self.km.fit_predict(data_pca)
        return labels

class TsneKm(Cluster):
    """
    TSNE(2) + Kmeans
    """

    def __init__(self, n_classes, use_log=False, name='tsne_km',
            metric='euclidean', use_exp=False, **params):
        super(TsneKm, self).__init__(n_classes, **params)
        self.use_log=use_log
        if metric=='kld':
            metric = symmetric_kld
        if 'k' in self.params:
            self.tsne = TSNE(self.params['k'], metric=metric)
        else:
            self.tsne = TSNE(2, metric=metric)
        self.km = KMeans(n_classes)
        self.name = name
        self.use_exp = use_exp

    def run(self, data):
        if sparse.issparse(data):
            data = data.toarray()
        if self.use_log:
            data = log1p(data)
        if self.use_exp:
            data = (10**data) - 1
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
        return y_pred.flatten()

class Bicluster(Cluster):
    """
    Spectral Biclustering
    """

    def __init__(self, n_classes, n_gene_classes=10, **params):
        super(Bicluster, self).__init__(n_classes, **params)
        self.n_gene_classes = n_gene_classes
        self.name = 'SpectralBicluster'

    def run(self, data):
        bc = SpectralBiclustering(n_clusters=(self.n_gene_classes, self.n_classes))
        bc.fit(data)
        gene_clusters = bc.row_labels_
        cell_clusters = bc.column_labels_
        return cell_clusters

class Cocluster(Cluster):
    """
    Spectral Coclustering
    """

    def __init__(self, n_classes, n_gene_classes=10, **params):
        super(Cocluster, self).__init__(n_classes, **params)
        self.n_gene_classes = n_gene_classes
        self.name = 'SpectralCocluster'

    def run(self, data):
        bc = SpectralCoclustering(n_clusters=(self.n_gene_classes, self.n_classes))
        bc.fit(data)
        gene_clusters = bc.row_labels_
        cell_clusters = bc.column_labels_
        return cell_clusters



def run_experiment(methods, data, n_classes, true_labels, n_runs=10, use_purity=True, use_nmi=False, use_ari=False, use_nne=False, consensus=False):
    """
    runs a pre-processing + clustering experiment...

    exactly one of use_purity, use_nmi, or use_ari can be true

    Args:
        methods: list of 2-tuples. The first element is either a single Preprocess object or a list of Preprocess objects, to be applied in sequence to the data. The second element is either a single Cluster object, a list of Cluster objects, or a list of lists, where each list is a sequence of Preprocess objects with the final element being a Cluster object.
        data: genes x cells array
        true_labels: 1d array of length cells
        consensus: if true, runs a consensus on cluster results for each method at the very end.
        use_purity, use_nmi, use_ari, use_nne: which error metric to use (at most one can be True)

    Returns:
        purities (list of lists)
        names (list of lists)
        other (dict): keys: timing, preprocessing, clusterings
    """
    results = []
    names = []
    clusterings = {}
    other_results = {}
    other_results['timing'] = {}
    other_results['preprocessing'] = {}
    if use_purity:
        purity_method = purity
    elif use_nmi:
        purity_method = nmi
    elif use_ari:
        purity_method = ari
    elif use_nne:
        purity_method = nne
    for i in range(n_runs):
        print('run {0}'.format(i))
        purities = []
        r = 0
        method_index = 0
        for preproc, cluster in methods:
            t0 = time.time()
            if isinstance(preproc, Preprocess):
                preprocessed, ll = preproc.run(data)
                output_names = preproc.output_names
            else:
                # if the input is a list, only use the first preproc result
                p1 = data
                output_names = ['']
                for p in preproc:
                    p1, ll = p.run(p1)
                    p1 = p1[0]
                    if output_names[0] != '':
                        output_names[0] = output_names[0] + '_' + p.output_names[0]
                    else:
                        output_names[0] = p.output_names[0]
                preprocessed = [p1]
            t1 = time.time() - t0
            for name, pre in zip(output_names, preprocessed):
                starting_index = method_index
                if isinstance(cluster, Cluster):
                    #try:
                        t0 = time.time()
                        labels = cluster.run(pre)
                        t2 = t1 + time.time() - t0
                        if use_nne:
                            purities.append(purity_method(pre, true_labels))
                        else:
                            purities.append(purity_method(labels, true_labels))
                        if i==0:
                            names.append(name + '_' + cluster.name)
                            clusterings[names[-1]] = []
                            other_results['timing'][names[-1]] = []
                        print(names[r])
                        clusterings[names[r]].append(labels)
                        print('time: ' + str(t2))
                        other_results['timing'][names[r]].append(t2)
                        print(purities[-1])
                        r += 1
                        method_index += 1
                    #except:
                    #    print('failed to do clustering')
                elif type(cluster) == list:
                    for c in cluster:
                        if isinstance(c, list):
                            t2 = t1
                            name2 = name
                            sub_data = pre.copy()
                            for subproc in c[:-1]:
                                t0 = time.time()
                                subproc_out, ll = subproc.run(sub_data)
                                sub_data = subproc_out[0]
                                name2 = name2 + '_' + subproc.output_names[0]
                                t2 += time.time() - t0
                            t0 = time.time()
                            labels = c[-1].run(sub_data)
                            t2 += time.time() - t0
                            if use_nne:
                                purities.append(purity_method(sub_data, true_labels))
                            else:
                                purities.append(purity_method(labels, true_labels))
                            if i==0:
                                names.append(name2 + '_' + c[-1].name)
                                clusterings[names[-1]] = []
                                other_results['timing'][names[-1]] = []
                            print(names[r])
                            clusterings[names[r]].append(labels)
                            other_results['timing'][names[r]].append(t2)
                            print('time: ' + str(t2))
                            print(purities[-1])
                            r += 1
                            method_index += 1
                        else:
                            try:
                                t0 = time.time()
                                labels = c.run(pre)
                                t2 = t1 + time.time() - t0
                                if i==0:
                                    names.append(name + '_' + c.name)
                                    clusterings[names[-1]] = []
                                    other_results['timing'][names[-1]] = []
                                if use_nne:
                                    purities.append(purity_method(pre, true_labels))
                                else:
                                    purities.append(purity_method(labels, true_labels))
                                print(names[r])
                                clusterings[names[r]].append(labels)
                                other_results['timing'][names[r]].append(t2)
                                print('time: ' + str(t2))
                                print(purities[-1])
                                r += 1
                                method_index += 1
                            except:
                                print('failed to do clustering')
                # find the highest purity for the pre-processing method
                # save the preprocessing result with the highest NMI
                num_clustering_results = method_index - starting_index
                clustering_results = purities[-num_clustering_results:]
                if i > 0 and len(clustering_results) > 0:
                    old_clustering_results = results[-1][starting_index:method_index]
                    if max(old_clustering_results) < max(clustering_results):
                        other_results['preprocessing'][name] = pre
                else:
                    other_results['preprocessing'][name] = pre
        print('\t'.join(names))
        print('purities: ' + '\t'.join(map(str, purities)))
        results.append(purities)
    consensus_purities = []
    if consensus:
        other_results['consensus'] = {}
        k = len(np.unique(true_labels))
        for name, clusts in clusterings.items():
            print(name)
            clusts = np.vstack(clusts)
            consensus_clust = CE.cluster_ensembles(clusts, verbose=False, N_clusters_max=k)
            other_results['consensus'][name] = consensus_clust
            if use_purity:
                consensus_purity = purity(consensus_clust.flatten(), true_labels)
                print('consensus purity: ' + str(consensus_purity))
                consensus_purities.append(consensus_purity)
            if use_nmi:
                consensus_nmi = nmi(true_labels, consensus_clust)
                print('consensus NMI: ' + str(consensus_nmi))
                consensus_purities.append(consensus_nmi)
            if use_ari:
                consensus_ari = ari(true_labels, consensus_clust)
                print('consensus ARI: ' + str(consensus_ari))
                consensus_purities.append(consensus_ari)
        print('consensus results: ' + '\t'.join(map(str, consensus_purities)))
    other_results['clusterings'] = clusterings
    return results, names, other_results

def generate_visualizations(methods, data, true_labels, base_dir = 'visualizations',
        figsize=(18,10), **scatter_options):
    """
    Generates visualization scatters for all the methods.

    Args:
        methods: follows same format as run_experiments. List of tuples.
        data: genes x cells
        true_labels: array of integers
        base_dir: base directory to save all the plots
        figsize: tuple of ints representing size of figure
        scatter_options: options for plt.scatter
    """
    plt.figure(figsize=figsize)
    for method in methods:
        preproc= method[0]
        if isinstance(preproc, Preprocess):
            preprocessed, ll = preproc.run(data)
            output_names = preproc.output_names
        else:
            # if the input is a list, only use the first preproc result
            p1 = data
            output_names = ['']
            for p in preproc:
                p1, ll = p.run(p1)
                p1 = p1[0]
                output_names[0] = output_names[0] + p.output_names[0]
            preprocessed = [p1]
        for r, name in zip(preprocessed, output_names):
            # TODO: cluster labels
            print(name)
            # if it's 2d, just display it... else, do tsne to reduce to 2d
            if r.shape[0]==2:
                r_dim_red = r
            else:
                # sometimes the data is too big to do tsne... (for sklearn)
                if sparse.issparse(r) and r.shape[0] > 100:
                    name = 'tsvd_' + name
                    tsvd = TruncatedSVD(50)
                    r_dim_red = tsvd.fit_transform(r.T)
                    try:
                        tsne = TSNE(2)
                        r_dim_red = tsne.fit_transform(r_dim_red).T
                        name = 'tsne_' + name
                    except:
                        tsvd2 = TruncatedSVD(2)
                        r_dim_red = tsvd2.fit_transform(r_dim_red).T
                else:
                    name = 'tsne_' + name
                    tsne = TSNE(2)
                    r_dim_red = tsne.fit_transform(r.T).T
            if isinstance(method[1], list):
                for clustering_method in method[1]:
                    try:
                        cluster_labels = clustering_method.run(r)
                    except:
                        print('clustering failed')
                        continue
                    output_path = base_dir + '/{0}_{1}_labels.png'.format(name, clustering_method.name)
                    visualize_dim_red(r_dim_red, cluster_labels, output_path, **scatter_options)
            else:
                clustering_method = method[1]
                try:
                    cluster_labels = clustering_method.run(r)
                except:
                    print('clustering failed')
                    continue
                output_path = base_dir + '/{0}_{1}_labels.png'.format(name, clustering_method.name)
                visualize_dim_red(r_dim_red, cluster_labels, output_path, **scatter_options)
            output_path = base_dir + '/{0}_true_labels.png'.format(name)
            visualize_dim_red(r_dim_red, true_labels, output_path, **scatter_options)
