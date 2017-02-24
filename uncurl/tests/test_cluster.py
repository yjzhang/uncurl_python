from unittest import TestCase

import numpy as np
from scipy.io import loadmat

import uncurl
from uncurl.simulation import generate_poisson_data

class ClusterTest(TestCase):

    def setUp(self):
        self.dat = loadmat('data/SCDE_k2_sup.mat')

    def test_kmeans_pp(self):
        data = self.dat['Dat']
        genes, cells = data.shape
        centers, assignments = uncurl.kmeans_pp(data, 3)
        self.assertEqual(centers.shape[0], genes)
        self.assertEqual(centers.shape[1], 3)
        # the center assignments are nondeterministic so...
        self.assertFalse(np.equal(centers[:,0], centers[:,1]).all())
        self.assertFalse(np.equal(centers[:,1], centers[:,2]).all())

    def test_cluster(self):
        data = self.dat['Dat']
        assignments, centers = uncurl.poisson_cluster(data, 3)
        # just checking that the values are valid
        self.assertFalse(np.isnan(centers).any())

    def test_simulation(self):
        """
        Basically this is to test that the Poisson EM can correctly separate
        clusters in simulated data.
        """
        centers = np.array([[0,10,20], [1, 11, 0], [50, 0, 100]])
        centers = centers.astype(float)
        data = generate_poisson_data(centers, 200)
        data = data.astype(float)
        assignments, c_centers = uncurl.poisson_cluster(data, 3)
        distances = np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                distances[i,j] = uncurl.poisson_dist(centers[:,i], c_centers[:,j])
        correspond = []
        for i in range(3):
            correspond.append(np.argmin(distances[i,:]))
            # assert that the learned clusters are close to the actual clusters
            self.assertTrue(min(distances[i,:]) < 4)
        self.assertFalse(correspond[0]==correspond[1])
        self.assertFalse(correspond[1]==correspond[2])
