import unittest
from unittest import TestCase
from flaky import flaky

import numpy as np

from scipy import sparse

import uncurl
from uncurl import pois_ll
from uncurl.simulation import generate_poisson_data

class PoissonTest(TestCase):

    def setUp(self):
        self.p1 = np.array([1.,2.,3.])
        self.p2 = np.array([2.,2.,3.])

    def test_poisson_dist(self):
        self.assertEqual(uncurl.poisson_dist(self.p1, self.p1), 0.0)
        self.assertEqual(uncurl.poisson_dist(self.p2, self.p2), 0.0)
        self.assertTrue(uncurl.poisson_dist(self.p1, self.p2) > 0.0)
        self.assertTrue(
                np.abs(uncurl.sparse_utils.poisson_dist(self.p1, self.p2) -
                    uncurl.poisson_dist(self.p1, self.p2)) < 1e-4)

    def test_sparse_poisson_dist(self):
        sp1 = sparse.csc_matrix(self.p1)
        sp2 = sparse.csc_matrix(self.p2)
        self.assertTrue(
                np.abs(uncurl.sparse_utils.poisson_dist(self.p1, self.p2) -
                    uncurl.poisson_dist(self.p1, self.p2)) < 1e-4)


    def test_poisson_ll(self):
        """
        Test Poisson log-likelihood
        """
        centers = np.array([[1,10,20], [1, 11, 1], [50, 1, 100]])
        centers = centers.astype(float)
        data, labs = generate_poisson_data(centers, 500)
        data = data.astype(float)
        starting_centers = centers
        poisson_ll = pois_ll.poisson_ll(data, starting_centers)
        p_isnan = np.isnan(poisson_ll)
        # just test that it's not nan
        self.assertFalse(p_isnan.any())

    def test_sparse_poisson_ll(self):
        """
        Test Poisson log-likelihood
        """
        centers = np.array([[0.1,10,20], [5, 15, 1], [50, 1, 0.1]])
        centers = centers.astype(float)
        data, labs = generate_poisson_data(centers, 500)
        data = data.astype(float)
        data = sparse.csc_matrix(data)
        starting_centers = centers
        poisson_ll = pois_ll.poisson_ll(data, starting_centers)
        p_isnan = np.isnan(poisson_ll)
        self.assertFalse(p_isnan.any())
        labels = poisson_ll.argmax(1)
        self.assertTrue((labels==labs).sum() >= 450)


