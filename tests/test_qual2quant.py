from __future__ import print_function

from unittest import TestCase

import numpy as np
from scipy import sparse
from scipy.io import loadmat

import uncurl

class Qual2QuantTest(TestCase):
    # TODO: test dataset

    def setUp(self):
        dat = loadmat('data/SCDE_test.mat')
        self.data = dat['dat'].toarray()[0:500, :]
        self.qualData = dat['M'].toarray()[0:500, :]


    def test_qual2quant(self):
        # simulated test data?
        # no... use M as a starting matrix
        # qual_matrix = np.zeros((self.data.shape[0], 2))
        starting_points = uncurl.qualNorm(self.data, self.qualData)
        self.assertTrue(starting_points.shape==(500, 2))
        self.assertFalse(np.isnan(starting_points).any())
        print((starting_points[:,0] == starting_points[:,1]).sum())
        self.assertTrue((starting_points[:,0] == starting_points[:,1]).sum() < 10)


    def test_qual2quant_sparse(self):
        # simulated test data?
        # no... use M as a starting matrix
        # qual_matrix = np.zeros((self.data.shape[0], 2))
        data_sparse = sparse.csc_matrix(self.data)
        starting_points = uncurl.qualNorm(data_sparse, self.qualData)
        self.assertTrue(starting_points.shape==(500, 2))
        self.assertFalse(np.isnan(starting_points).any())
        print((starting_points[:,0] == starting_points[:,1]).sum())
        self.assertTrue((starting_points[:,0] == starting_points[:,1]).sum() < 10)


    def test_qual2quant_missing_data(self):
        # simulated test data?
        # no... use M as a starting matrix
        # qual_matrix = np.zeros((self.data.shape[0], 2))
        qualData_m = self.qualData.copy()
        for i in range(300):
            qualData_m[i,:] = -1
        starting_points = uncurl.qualNorm(self.data, qualData_m)
        self.assertTrue(starting_points.shape==(500, 2))
        self.assertFalse(np.isnan(starting_points).any())
        print((starting_points[:,0] == starting_points[:,1]).sum())
        self.assertTrue((starting_points[:,0] == starting_points[:,1]).sum() < 10)


