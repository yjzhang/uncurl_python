from __future__ import print_function

import unittest
from unittest import TestCase

import numpy as np

from uncurl import nb_state_estimation, simulation
from uncurl.evaluation import purity

@unittest.skip('nb methods currently not supported')
class StateEstimationTest(TestCase):

    def setUp(self):
        pass

    def test_random_1(self):
        """
        Test NB state estimation with random parameters
        """
        M, W, R = simulation.generate_nb_states(2, 200, 20)
        data = simulation.generate_nb_state_data(M, W, R)
        M_noised = M + 0.1*(np.random.random(M.shape)-0.5)
        M_, W_, R_, ll = nb_state_estimation.nb_estimate_state(data, 2, init_means=M_noised, R = R, disp=False)
        c1 = W.argmax(0)
        c2 = W_.argmax(0)
        p = purity(c2, c1)
        print(p)
        print(data)
        print(M)
        print(M_)
        self.assertTrue(p > 0.7)
