from unittest import TestCase

import numpy as np
from scipy.io import loadmat

import uncurl

class Qual2QuantTest(TestCase):

    def setUp(self):
        self.dat = loadmat('data/SCDE_k2_sup.mat')

    def test_qual2quant(self):
        pass
