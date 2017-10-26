from unittest import TestCase

import numpy as np

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

    def test_poisson_ll(self):
        """
        """
        pois_ll.poisson_ll_2(self.p1, self.p2)

    def test_zip_ll(self):
        centers = np.array([[1,10,20], [1, 11, 1], [50, 1, 100]])
        centers = centers.astype(float)
        data, labs = generate_poisson_data(centers, 500)
        data = data.astype(float)
        starting_centers = centers
        starting_L = np.array([[0,0,0], [0, 0, 0], [0, 0, 0]])
        starting_L = starting_L.astype(float)
        zip_ll = pois_ll.zip_ll(data, starting_centers, starting_L)
        poisson_ll = pois_ll.poisson_ll(data, starting_centers)
        #self.assertTrue((zip_ll<=poisson_ll).all())
        zip_ll2 = pois_ll.zip_ll(data, starting_centers + 1.0, starting_L)
        p_isnan = np.isnan(poisson_ll)
        pll = poisson_ll[p_isnan]
        zll2 = zip_ll2[p_isnan]
        self.assertTrue((zll2<=pll).all())
