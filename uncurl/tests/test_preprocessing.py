from unittest import TestCase

import numpy as np
from scipy.io import loadmat
from scipy import sparse

import uncurl
from uncurl.preprocessing import sparse_var
from uncurl.simulation import generate_poisson_data
from uncurl.evaluation import purity

class PreprocessingTest(TestCase):

    def setUp(self):
        dat = loadmat('data/SCDE_k2_sup.mat')
        self.data_sparse = sparse.csc_matrix(dat['Dat'])
        self.data_dense = dat['Dat']
        self.labs = dat['Lab'].flatten()

    def testSparseVar(self):
        """
        Test sparse variance
        """
        dense_var = np.var(self.data_dense, 1)
        sp_var = sparse_var(self.data_sparse)
        se = np.sqrt(np.sum((sp_var - dense_var)**2))
        print(se)
        self.assertTrue(se < 1e-6)

    def testMaxVarGenes(self):
        """
        test max variance genes for dense and sparse matrices
        """
        n_genes =self.data_sparse.shape[0]
        genes1 = uncurl.max_variance_genes(self.data_dense, nbins=1, frac=0.5)
        genes2 = uncurl.max_variance_genes(self.data_sparse, nbins=1, frac=0.5)
        self.assertEqual(set(genes1), set(genes2))
        self.assertEqual(len(genes1), int(0.5*n_genes))
        genes1 = uncurl.max_variance_genes(self.data_dense, nbins=5, frac=0.2)
        genes2 = uncurl.max_variance_genes(self.data_sparse, nbins=5, frac=0.2)
        self.assertEqual(set(genes1), set(genes2))
        self.assertEqual(len(genes1), 5*int((n_genes/5)*0.2))

