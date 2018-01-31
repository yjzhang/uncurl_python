from __future__ import print_function

from unittest import TestCase
from flaky import flaky

import numpy as np
from scipy.io import loadmat
from scipy import sparse

import uncurl
from uncurl.simulation import generate_poisson_data
from uncurl.evaluation import purity

@flaky
class SparseClusterTest(TestCase):

    def setUp(self):
        dat = loadmat('data/SCDE_k2_sup.mat')
        self.data = sparse.csc_matrix(dat['Dat'])
        self.labs = dat['Lab'].flatten()

    def test_kmeans_pp(self):
        data = self.data
        genes, cells = data.shape
        centers, assignments = uncurl.kmeans_pp(data, 3)
        self.assertEqual(centers.shape[0], genes)
        self.assertEqual(centers.shape[1], 3)
        # the center assignments are nondeterministic so...
        self.assertFalse(np.equal(centers[:,0], centers[:,1]).all())
        self.assertFalse(np.equal(centers[:,1], centers[:,2]).all())

    def test_cluster(self):
        data = self.data
        assignments, centers = uncurl.poisson_cluster(data, 2)
        self.assertEqual(assignments.shape[0], data.shape[1])
        self.assertEqual(centers.shape[0], data.shape[0])
        # just checking that the values are valid
        self.assertFalse(np.isnan(centers).any())
        self.assertTrue(purity(assignments, self.labs) > 0.8)

    def test_simulation(self):
        """
        Basically this is to test that the Poisson EM can correctly separate
        clusters in simulated data.
        """
        centers = np.array([[1,10,20], [1, 11, 1], [50, 1, 100]])
        centers = centers.astype(float)
        data, labs = generate_poisson_data(centers, 500)
        data = data.astype(float)
        data = sparse.csc_matrix(data)
        assignments, c_centers = uncurl.poisson_cluster(data, 3)
        distances = np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                distances[i,j] = uncurl.poisson_dist(centers[:,i], c_centers[:,j])
        print(assignments)
        print(labs)
        print(purity(assignments, labs))
        self.assertTrue(purity(assignments, labs) > 0.65)

