from __future__ import print_function

import unittest
from unittest import TestCase
from flaky import flaky

import numpy as np
from scipy.io import loadmat

import uncurl
from uncurl.simulation import generate_poisson_data, generate_zip_data
from uncurl.evaluation import purity
from uncurl.zip_clustering import zip_fit_params_mle

@flaky(max_runs=3)
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
        self.assertEqual(assignments.shape[0], data.shape[1])
        self.assertEqual(centers.shape[0], data.shape[0])
        # just checking that the values are valid
        self.assertFalse(np.isnan(centers).any())

    def test_simulation(self):
        """
        Basically this is to test that the Poisson EM can correctly separate
        clusters in simulated data.
        """
        centers = np.array([[1,10,20], [1, 11, 1], [50, 1, 100]])
        centers = centers.astype(float)
        data, labs = generate_poisson_data(centers, 500)
        data = data.astype(float)
        assignments, c_centers = uncurl.poisson_cluster(data, 3)
        distances = np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                distances[i,j] = uncurl.poisson_dist(centers[:,i], c_centers[:,j])
        self.assertTrue(purity(assignments, labs) > 0.8)

    @flaky(max_runs=3)
    @unittest.skip('zip methods are unsupported')
    def test_zip_simulation(self):
        """
        ZIP clustering on poisson-simulated data
        """
        centers = np.array([[0.1,10,20], [0.1, 11, 0.1], [50, 0.1, 100]])
        centers = centers.astype(float)
        data, labs = generate_poisson_data(centers, 500)
        data = data.astype(float)
        assignments, c_centers, c_zeros = uncurl.zip_cluster(data, 3)
        self.assertTrue(purity(assignments, labs) > 0.8)

    @flaky(max_runs=3)
    @unittest.skip('zip methods are unsupported')
    def test_zip_fit(self):
        """
        Tests the algorithm for fitting a ZIP distribution.
        """
        for i in range(10):
            centers = np.random.randint(10, 1000, (3,1))
            M = np.random.random((3,1))
            data, labs = generate_zip_data(centers, M, 300)
            L_, M_ = zip_fit_params_mle(data)
            self.assertFalse(np.isnan(L_).any())
            self.assertFalse(np.isnan(M_).any())
            self.assertFalse(np.isnan(L_).any())
            self.assertFalse(np.isnan(M_).any())
            self.assertTrue(np.mean(np.abs(M.flatten() - M_)) < 0.2)
            self.assertTrue(np.mean(np.abs(centers.flatten() - L_)) < 10)

    @flaky(max_runs=3)
    @unittest.skip('zip methods are unsupported')
    def test_zip_simulation_2(self):
        """
        ZIP clustering on ZIP-simulated data
        """
        centers = np.random.randint(10, 1000, (3,3))
        L = np.random.random((3,3))
        print(centers)
        print(L)
        centers = centers.astype(float)
        data, labs = generate_zip_data(centers, L, 1000)
        data = data.astype(float)
        print(data)
        assignments, c_centers, c_zeros = uncurl.zip_cluster(data, 3)
        distances = np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                distances[i,j] = uncurl.poisson_dist(centers[:,i], c_centers[:,j])
        print(c_centers)
        print(c_zeros)
        print(purity(assignments, labs))
        self.assertTrue(purity(assignments, labs) > 0.6)
        #self.assertFalse(correspond[0]==correspond[1])
        #self.assertFalse(correspond[1]==correspond[2])
