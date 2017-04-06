from unittest import TestCase

import numpy as np

import uncurl
from uncurl import pois_ll

class PoissonTest(TestCase):

    def setUp(self):
        self.p1 = np.array([1.,2.,3.])
        self.p2 = np.array([2.,2.,3.])

    def test_poisson_dist(self):
        self.assertEqual(uncurl.poisson_dist(self.p1, self.p1), 0.0)
        self.assertEqual(uncurl.poisson_dist(self.p2, self.p2), 0.0)
        self.assertTrue(uncurl.poisson_dist(self.p1, self.p2) > 0.0)

    def test_poisson_ll(self):
        """
        """
        pois_ll.poisson_ll_2(self.p1, self.p2)
