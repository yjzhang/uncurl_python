from unittest import TestCase

import numpy as np
from scipy.io import loadmat

import uncurl

class Qual2QuantTest(TestCase):
    # TODO: test dataset

    def setUp(self):
        dat = loadmat('data/SCDE_test.mat')
        self.data = dat['dat'].toarray()
        self.qualData = dat['M'].toarray()


    def test_qual2quant(self):
        # simulated test data?
        # no... use M as a starting matrix
        # qual_matrix = np.zeros((self.data.shape[0], 2))
        starting_points = uncurl.qualNorm(self.data, self.qualData)
        self.assertTrue(starting_points.shape==(2904, 2))
        self.assertFalse(np.isnan(starting_points).any())
        self.assertFalse((starting_points == 0).any())
