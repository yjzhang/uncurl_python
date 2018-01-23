from __future__ import print_function

from unittest import TestCase

import numpy as np
from scipy import sparse
from scipy.io import loadmat

import uncurl

class ExperimentRunnerTest(TestCase):
    # TODO: test dataset

    def setUp(self):
        dat = loadmat('data/SCDE_test.mat')
        self.data = dat['dat'].toarray()[0:500, :]
        self.data = sparse.csc_matrix(self.data)
        self.labs = dat['Lab'][0]

    def test_run(self):
        se = uncurl.experiment_runner.PoissonSE(clusters=2)
        results, ll = se.run(self.data)
        self.assertTrue(len(results)==1)
        self.assertTrue(results[0].shape[0]==2)

    def test_runExperiment(self):
        se = uncurl.experiment_runner.PoissonSE(clusters=2, max_iters=10, inner_max_iters=50)
        argmax = uncurl.experiment_runner.Argmax(n_classes=2)
        km = uncurl.experiment_runner.KM(n_classes=2)
        methods = [(se, [argmax, km])]
        results, names, other = uncurl.experiment_runner.run_experiment(methods, self.data, 2, self.labs, n_runs=2)
        self.assertEqual(len(results), 2)
        self.assertTrue('clusterings' in other)
        self.assertTrue('timing' in other)
        self.assertTrue('preprocessing' in other)
        print(results)
        self.assertTrue(results[0][0]>0.95)

    def test_runExperiment_2(self):
        se = uncurl.experiment_runner.PoissonSE(clusters=2, max_iters=10, inner_max_iters=50)
        pre = uncurl.experiment_runner.Preprocess()
        argmax = uncurl.experiment_runner.Argmax(n_classes=2)
        km = uncurl.experiment_runner.KM(n_classes=2)
        pca_km = uncurl.experiment_runner.PcaKm(k=8, n_classes=2)
        methods = [(se, [argmax, km]), (pre, [km, pca_km])]
        results, names, other = uncurl.experiment_runner.run_experiment(methods, self.data, 2, self.labs, n_runs=2)
        self.assertEqual(len(results), 2)
        self.assertTrue('clusterings' in other)
        self.assertTrue('timing' in other)
        self.assertTrue('preprocessing' in other)
        self.assertTrue(results[0][0]>0.95)
