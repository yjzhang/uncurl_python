from __future__ import print_function

from unittest import TestCase

import numpy as np
from scipy import sparse
from scipy.io import loadmat

import uncurl

class RealDataTest(TestCase):
    """
    tests results on actual datasets: 10x_pooled,
    """

    def setUp(self):
        dat = loadmat('data/10x_pooled_400.mat')
        self.data = sparse.csc_matrix(dat['data'])
        self.labs = dat['labels'].flatten()

    def test_runSEExperiment(self):
        # gene selection
        genes = uncurl.max_variance_genes(self.data)
        data_subset = self.data[genes,:]
        se = uncurl.experiment_runner.PoissonSE(clusters=8, max_iters=10, inner_max_iters=100)
        argmax = uncurl.experiment_runner.Argmax(n_classes=8)
        km = uncurl.experiment_runner.KM(n_classes=8)
        methods = [(se, [argmax, km])]
        results, names, other = uncurl.experiment_runner.run_experiment(methods, data_subset, 8, self.labs, n_runs=1,
                use_purity=False, use_nmi=True)
        self.assertEqual(len(results), 1)
        self.assertTrue('clusterings' in other)
        self.assertTrue('timing' in other)
        self.assertTrue('preprocessing' in other)
        print(results)
        # NMI should be > 0.8 on 10x_pure_pooled
        self.assertTrue(results[0][0]>0.8)
        self.assertTrue(results[0][0]>0.8)
