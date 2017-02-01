from unittest import TestCase

import numpy as np
from scipy.io import loadmat

import uncurl

class ClusterTest(TestCase):

    def setUp(self):
        self.dat = loadmat('data/SCDE_k2_sup.mat')

    def test_kmeans_pp(self):
        data = self.dat['Dat']
        genes, cells = data.shape
        centers = uncurl.kmeans_pp(data, 3)
        self.assertEqual(centers.shape[0], genes)
        self.assertEqual(centers.shape[1], 3)
        # the center assignments are nondeterministic so...
        self.assertFalse(np.equal(centers[:,0], centers[:,1]).all())
        self.assertFalse(np.equal(centers[:,1], centers[:,2]).all())

    def test_cluster(self):
        data = self.dat['Dat']
        assignments, centers = uncurl.poisson_cluster(data, 3)
