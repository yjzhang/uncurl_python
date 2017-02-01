from unittest import TestCase

import numpy as np
from scipy.io import loadmat

import uncurl

class ClusterTest(TestCase):

    def setUp(self):
        self.dat = loadmat('data/SCDE_k2_sup.mat')

    def test_poisson_ll(self):
        pass
