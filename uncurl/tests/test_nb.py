from unittest import TestCase

import numpy as np

import uncurl

from uncurl import nb_cluster, simulation

class NBTest(TestCase):

    def setUp(self):
        self.p1 = np.array([1.,2.,3.])
        self.p2 = np.array([2.,2.,3.])

    def test_negative_binomial(self):
        """
        Test NB log-likelihood, other helper functions
        """
        P = np.array([[0.5,0.1,0.8],
                      [0.5,0.2,0.7],
                      [0.1,0.3,0.9]])
        R = np.array([[1.,1.,1.],
                      [2.,2.,2.],
                      [2.,2.,2.]])
        data = simulation.generate_nb_data(P, R, 40)
        data = data.astype(float)
        data += 1e-8
        ll = nb_cluster.nb_ll(data, P, R)
        self.assertEqual(ll.shape, (40,3))
        # test derivative
        d1 = nb_cluster._r_deriv(R[:,0], P[:,0], data)
        self.assertEqual(d1.shape, (3,))
        # test nb cluster
        # how to test the results... they're often not good...
        p,r,a = nb_cluster.nb_cluster(data,3)
        self.assertEqual(p.shape, P.shape)
        self.assertEqual(r.shape, R.shape)


    def test_nb_fit(self):
        """
        Tests fitting an NB distribution
        """
        P = np.array([[0.5],
                      [0.9],
                      [0.1]])
        R = np.array([[1.],
                      [5.],
                      [2.]])
        data = simulation.generate_nb_data(P, R, 20)
        p, r = nb_cluster.nb_fit(data)
