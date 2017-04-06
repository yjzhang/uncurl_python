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

        7 cells, 4 genes, 2 clusters
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
        sim_means_noised = sim_means + 5*(np.random.random(sim_means.shape)-0.5)
        m, w = state_estimation.poisson_estimate_state(sim_data, 2, init_means=sim_means_noised, max_iters=10, disp=False)
        print m
        print w
        print w.sum(0)
        self.assertTrue(np.max(w.sum(0) - 1.0)<0.01)
        # mean error in M is less than 5
        self.assertTrue(np.mean(np.abs(sim_means-m))<10.0)
        # mean error in W is less than 0.2 (arbitrary boundary)
        self.assertTrue(np.mean(np.abs(sim_assignments-w))<0.3)

    def test_state_estimation_2(self):
        """
        Generate sample data from a slightly larger set to see that the state
        estimation is accurate.

        11 cells, 5 genes, 3 clusters

        This might fail due to inherent randomness...
        """
        sim_means = np.array([[20.,30.,4.],
                              [10.,3.,9.],
                              [90.,50.,10.],
                              [10.,4.,30.],
                              [35.,10.,2.]])
        sim_assignments = np.array([[0.1,0.2,0.3,0.4,0.1,0.7,0.6,0.9,0.5,0.2,0.1],
                                    [0.6,0.7,0.3,0.4,0.1,0.2,0.1,0.1,0.0,0.3,0.8],
                                    [0.3,0.1,0.4,0.2,0.8,0.1,0.3,0.0,0.5,0.5,0.1]])
        sim_data = simulation.generate_state_data(sim_means, sim_assignments)
        sim_data = sim_data + 1e-8
        print sim_data
        # add noise to the mean
        sim_means_noised = sim_means + 5*(np.random.random(sim_means.shape)-0.5)
        m, w = state_estimation.poisson_estimate_state(sim_data, 3, init_means=sim_means_noised, max_iters=10, disp=False)
        print m
        print w
        print w.sum(0)
        self.assertTrue(np.max(w.sum(0) - 1.0)<0.01)
        # mean error in M is less than 10
        self.assertTrue(np.mean(np.abs(sim_means-m))<10.0)
        # mean error in W is less than 0.4 (arbitrary boundary)
        self.assertTrue(np.mean(np.abs(sim_assignments-w))<0.4)

    def test_random_means(self):
        """
        Test state estimation with random means and weights.

        200 cells, 20 genes, 2 clusters
        """
        sim_m, sim_w = simulation.generate_poisson_states(2, 200, 20)
        sim_data = simulation.generate_state_data(sim_m, sim_w)
        sim_means_noised = sim_m + 5*(np.random.random(sim_m.shape)-0.5)
        m, w = state_estimation.poisson_estimate_state(sim_data, 2, init_means=sim_means_noised, max_iters=10, disp=False)
        self.assertTrue(np.max(w.sum(0) - 1.0)<0.001)
        self.assertTrue(np.mean(np.abs(sim_m-m))<50.0)
        self.assertTrue(np.mean(np.abs(sim_w-w))<0.4)

    def test_random_means_2(self):
        """
        Test state estimation with random means and weights.

        20 cells, 200 genes, 2 clusters
        """
        sim_m, sim_w = simulation.generate_poisson_states(2, 20, 200)
        sim_data = simulation.generate_state_data(sim_m, sim_w)
        sim_means_noised = sim_m + 5*(np.random.random(sim_m.shape)-0.5)
        m, w = state_estimation.poisson_estimate_state(sim_data, 2, init_means=sim_means_noised, max_iters=10, disp=False)
        self.assertTrue(np.max(w.sum(0) - 1.0)<0.001)
        self.assertTrue(np.mean(np.abs(sim_m-m))<60.0)
        self.assertTrue(np.mean(np.abs(sim_w-w))<0.5)
