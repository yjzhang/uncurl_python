from __future__ import print_function

from unittest import TestCase

import numpy as np
from scipy import sparse
from scipy.io import loadmat

import uncurl

class RealDataTest(TestCase):
    """
    tests results on actual datasets: 10x_pooled, Zeisel 7-cluster subset,
    maybe add others?
    """

    def setUp(self):
        dat = loadmat('data/10x_pooled_400.mat')
        self.data = sparse.csc_matrix(dat['data'])
        self.labs = dat['labels'].flatten()
        dat_z = loadmat('data/GSE60361_dat.mat')
        self.data_z = sparse.csc_matrix(dat_z['Dat'])
        self.labs_z = dat_z['ActLabs'].flatten()

    def test_10xSE(self):
        # gene selection
        genes = uncurl.max_variance_genes(self.data)
        data_subset = self.data[genes,:]
        # smaller # of iterations than default so it finishes faster...
        se = uncurl.experiment_runner.PoissonSE(clusters=8, max_iters=10,
                inner_max_iters=80)
        argmax = uncurl.experiment_runner.Argmax(n_classes=8)
        km = uncurl.experiment_runner.KM(n_classes=8)
        methods = [(se, [argmax, km])]
        results, names, other = uncurl.experiment_runner.run_experiment(
                methods, data_subset, 8, self.labs, n_runs=1,
                use_purity=False, use_nmi=True)
        print(results)
        # NMI should be > 0.75 on 10x_pure_pooled 
        # (accounting for lower than default iter count)
        self.assertTrue(results[0][0]>0.75)
        self.assertTrue(results[0][1]>0.75)

    def test_Zeisel(self):
        # gene selection
        genes = uncurl.max_variance_genes(self.data_z)
        data_subset = self.data_z[genes,:]
        # smaller # of iterations than default so it finishes faster...
        se = uncurl.experiment_runner.PoissonSE(clusters=7, max_iters=10,
                inner_max_iters=80)
        argmax = uncurl.experiment_runner.Argmax(n_classes=7)
        km = uncurl.experiment_runner.KM(n_classes=7)
        methods = [(se, [argmax, km])]
        results, names, other = uncurl.experiment_runner.run_experiment(
                methods, data_subset, 7, self.labs_z, n_runs=1,
                use_purity=False, use_nmi=True)
        print(results)
        # NMI should be > 0.75 on Zeisel subset as well
        self.assertTrue(results[0][0]>0.75)
        self.assertTrue(results[0][1]>0.75)
