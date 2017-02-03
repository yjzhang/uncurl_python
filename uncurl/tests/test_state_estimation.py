from unittest import TestCase

import numpy as np
from scipy.io import loadmat

from uncurl import state_estimation, simulation

class StateEstimationTest(TestCase):

    def setUp(self):
        pass

    def test_state_estimation(self):
        """
        Generate sample data from a small set to see that the state
        estimation is accurate.
        """
        sim_means = np.array([[20.,30.],
                              [10.,3.],
                              [90.,50.],
                              [10.,4.]])
        sim_assignments = np.array([[0.1,0.2,0.3,0.4,0.5,0.8,0.9],
                                    [0.9,0.8,0.7,0.6,0.5,0.2,0.1]])
        sim_data = simulation.generate_state_data(sim_means, sim_assignments)
        sim_data = sim_data + 1e-8
        print sim_data
        # add noise to the mean
        sim_means += 2*(np.random.random(sim_means.shape)-0.5)
        m, w = state_estimation.poisson_estimate_state(sim_data, sim_means, max_iters=10)
        print m
        print w
        # mean error in M is less than 5
        self.assertTrue(np.mean(np.abs(sim_means-m))<5.0)
        # mean error in W is less than 0.2 (arbitrary boundary)
        self.assertTrue(np.mean(np.abs(sim_assignments-w))<0.2)
