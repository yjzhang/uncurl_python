from unittest import TestCase

import numpy as np
from scipy.io import loadmat

from uncurl import state_estimation

class StateEstimationTest(TestCase):

    def setUp(self):
        self.dat = loadmat('data/SCDE_k2_sup.mat')

    def test_state_estimation(self):
        pass
