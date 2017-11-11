import unittest
from unittest import TestCase
from flaky import flaky

import numpy as np

from scipy.io import loadmat
from scipy import sparse

import uncurl
from uncurl.simulation import generate_poisson_data
from uncurl import fit_dist_data

@flaky(max_runs=4)
class FitDistTest(TestCase):

    def setUp(self):
        pass

    def testPoissonData(self):
        """
        Test with generated unimodal Poisson dataset.
        """
        centers = np.array([[1], [10], [50]])
        centers = centers.astype(float)
        data, labs = generate_poisson_data(centers, 500)
        fit_errors = fit_dist_data.DistFitDataset(data)
        self.assertTrue((fit_errors['poiss'] < fit_errors['norm']).all())
        self.assertTrue((fit_errors['poiss'] < fit_errors['lognorm']).all())

    def testNormalData(self):
        """
        Test with generated unimodal Normal dataset.
        """
        centers = np.array([[100], [20], [50]])
        variances = np.array([[1.0], [1.0], [5.0]])
        centers = centers.astype(float)
        data = np.random.normal(centers, variances, size=(3,500))
        fit_errors = fit_dist_data.DistFitDataset(data)
        self.assertTrue((fit_errors['poiss'] > fit_errors['norm']).all())
        self.assertTrue((fit_errors['norm'] < fit_errors['lognorm']).all())

    @unittest.skip('still working on this')
    def testLogNormalData(self):
        """
        Test with generated unimodal Log-Normal dataset.
        """
        centers = np.array([[-1.0], [0.0], [-2]])
        variances = np.array([[2.0], [1.2], [1.5]])
        centers = centers.astype(float)
        data = np.random.lognormal(centers, variances, size=(3,500))
        print(data.round())
        print(data.round().max(1))
        fit_errors = fit_dist_data.DistFitDataset(data)
        print(fit_errors)
        self.assertTrue((fit_errors['poiss'] > fit_errors['lognorm']).all())
        self.assertTrue((fit_errors['norm'] > fit_errors['lognorm']).all())

