from __future__ import print_function

import unittest
from unittest import TestCase
from flaky import flaky

import numpy as np

from uncurl import nb_cluster, simulation
from uncurl.nb_clustering import nb_ll, nb_fit
from uncurl.evaluation import purity


@flaky
@unittest.skip('nb methods currently not supported')
class NBTest(TestCase):

    def setUp(self):
        self.p1 = np.array([1.,2.,3.])
        self.p2 = np.array([2.,2.,3.])

    def test_negative_binomial(self):
        """
        Test NB log-likelihood, nb_cluster
        """
        P = np.array([[0.5,0.4,0.8],
                      [0.5,0.3,0.7],
                      [0.5,0.3,0.9]])
        R = np.array([[1.,8.,10.],
                      [2.,8.,24],
                      [3.,6.,30.]])
        data, labels = simulation.generate_nb_data(P, R, 100)
        data = data.astype(float)
        #data += 1e-8
        ll = nb_ll(data, P, R)
        self.assertEqual(ll.shape, (100,3))
        self.assertFalse(np.isnan(ll).any())
        self.assertFalse(np.isinf(ll).any())
        # test derivative
        # test nb cluster
        # how to test the results... they're often not good...
        a,p,r = nb_cluster(data,3)
        self.assertEqual(p.shape, P.shape)
        self.assertEqual(r.shape, R.shape)
        p_nans = np.isnan(p)
        r_nans = np.isnan(r)
        self.assertFalse(p_nans.any())
        self.assertFalse(r_nans.any())
        # assert that all the points aren't being put into
        # the same cluster.
        self.assertTrue(purity(labels, a) > 0.8)
        self.assertFalse((a==a[0]).all())


    def test_nb_fit(self):
        """
        Tests fitting an NB distribution
        """
        P = np.array([[0.5],
                      [0.3],
                      [0.4]])
        R = np.array([[1.],
                      [8.],
                      [2.]])
        data, _ = simulation.generate_nb_data(P, R, 500)
        p, r = nb_fit(data)
        p_nans = np.isnan(p)
        r_nans = np.isnan(r)
        self.assertFalse(p_nans.any())
        self.assertFalse(r_nans.any())
        self.assertFalse(np.isinf(p).any())
        self.assertFalse(np.isinf(r).any())
        self.assertTrue(np.sum(np.abs(p - P.flatten())**2)/3 < 0.5)
        print(r)
        print(np.sqrt(np.sum(np.abs(r - R.flatten())**2))/3)
        self.assertTrue(np.sqrt(np.sum(np.abs(r - R.flatten())**2))/3 < 3)

    def test_nb_fit_random(self):
        """
        Tests fitting an NB distribution with random parameters
        """
        for i in range(5):
            P = np.random.random((3,1))*0.9+0.1
            R = np.random.randint(1, 100, (3,1))
            data, _ = simulation.generate_nb_data(P, R, 500)
            try:
                p, r = nb_fit(data)
            except ValueError:
                continue
            p_nans = np.isnan(p)
            r_nans = np.isnan(r)
            print(P)
            print(R)
            print(p)
            print(r)
            print(np.sqrt(np.sum(np.abs(r - R.flatten())**2))/3)
            self.assertTrue(np.sqrt(np.sum(np.abs(r - R.flatten())**2))/3 < 35)
            self.assertFalse(p_nans.any())
            self.assertFalse(r_nans.any())
            self.assertFalse(np.isinf(p).any())
            self.assertFalse(np.isinf(r).any())
            self.assertTrue(np.sum(np.abs(p - P.flatten())**2)/3 < 0.5)

