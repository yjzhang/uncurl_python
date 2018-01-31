from __future__ import print_function

import itertools

from unittest import TestCase
from flaky import flaky

import numpy as np
from scipy import sparse

from uncurl import state_estimation, simulation
from uncurl.nolips import objective, sparse_objective

class SparseStateEstimationTest(TestCase):

    def setUp(self):
        pass

    @flaky
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
        sim_data = sparse.csc_matrix(sim_data)
        print(sim_data)
        # add noise to the mean
        sim_means_noised = sim_means + 5*(np.random.random(sim_means.shape)-0.5)
        m, w, ll = state_estimation.poisson_estimate_state(sim_data, 2, init_means=sim_means_noised, max_iters=10, disp=False)
        print(m)
        print(w)
        self.assertTrue(np.max(w.sum(0) - 1.0)<0.01)
        # mean error in M is less than 5
        self.assertTrue(np.mean(np.abs(sim_means-m))<10.0 or
                np.mean(np.abs(sim_means-m[:,[1,0]]))<10.0)
        # mean error in W is less than 0.2 (arbitrary boundary)
        self.assertTrue(np.mean(np.abs(sim_assignments-w))<0.3 or
                np.mean(np.abs(sim_assignments-w[[1,0],:]))<0.3)

    @flaky
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
        sim_data = sparse.csc_matrix(sim_data)
        print(sim_data)
        # add noise to the mean
        sim_means_noised = sim_means + 5*(np.random.random(sim_means.shape)-0.5)
        m, w, ll = state_estimation.poisson_estimate_state(sim_data, 3, init_means=sim_means_noised, max_iters=10, disp=False, parallel=False)
        print(m)
        print(w)
        print(w.sum(0))
        self.assertTrue(np.max(w.sum(0) - 1.0)<0.01)
        # mean error in M is less than 10
        means_good = False
        weights_good = False
        # test every permutation of clusters
        for p in itertools.permutations([0,1,2]):
            means_good = means_good or (np.mean(np.abs(sim_means-m[:,p]))<10.0)
            weights_good = weights_good or (np.mean(np.abs(sim_assignments-w[p,:]))<0.2)
        self.assertTrue(means_good)
        self.assertTrue(weights_good)

    def test_random_means_lbfgs(self):
        """
        Test state estimation with random means and weights.

        200 cells, 20 genes, 2 clusters
        """
        sim_m, sim_w = simulation.generate_poisson_states(2, 200, 20)
        sim_data = simulation.generate_state_data(sim_m, sim_w)
        sim_data = sparse.csc_matrix(sim_data)
        #sim_means_noised = sim_m + 5*(np.random.random(sim_m.shape)-0.5)
        m, w, ll = state_estimation.poisson_estimate_state(sim_data, 2, max_iters=10, disp=False, method='L-BFGS-B')
        self.assertTrue(np.max(w.sum(0) - 1.0)<0.001)
        means_good = False
        weights_good = False
        for p in itertools.permutations([0,1]):
            means_good = means_good or (np.mean(np.abs(sim_m-m[:,p]))<20.0)
            weights_good = weights_good or (np.mean(np.abs(sim_w-w[p,:]))<0.3)
        self.assertTrue(means_good)
        self.assertTrue(weights_good)

    def test_random_means_km_init(self):
        """
        Test state estimation with random means and weights.

        200 cells, 20 genes, 2 clusters
        """
        sim_m, sim_w = simulation.generate_poisson_states(2, 200, 20)
        sim_data = simulation.generate_state_data(sim_m, sim_w)
        sim_data = sparse.csc_matrix(sim_data)
        m, w, ll = state_estimation.poisson_estimate_state(sim_data, 2, max_iters=10, disp=False, initialization='km')
        self.assertTrue(np.max(w.sum(0) - 1.0)<0.001)
        means_good = False
        weights_good = False
        for p in itertools.permutations([0,1]):
            means_good = means_good or (np.mean(np.abs(sim_m-m[:,p]))<20.0)
            weights_good = weights_good or (np.mean(np.abs(sim_w-w[p,:]))<0.3)
        self.assertTrue(means_good)
        self.assertTrue(weights_good)

    def test_random_means_tsvd_init(self):
        """
        Test state estimation with random means and weights.

        200 cells, 20 genes, 2 clusters
        """
        sim_m, sim_w = simulation.generate_poisson_states(2, 200, 20)
        sim_data = simulation.generate_state_data(sim_m, sim_w)
        sim_data = sparse.csc_matrix(sim_data)
        m, w, ll = state_estimation.poisson_estimate_state(sim_data, 2, max_iters=10, disp=False, initialization='tsvd', threads=1)
        self.assertTrue(np.max(w.sum(0) - 1.0)<0.001)
        obj = sparse_objective(sim_data.data,
                sim_data.indices,
                sim_data.indptr,
                200, 20, m, w)
        self.assertEqual(ll, obj)
        dense_obj = objective(sim_data.toarray(), m, w)
        self.assertTrue(np.abs(obj-dense_obj) < 1e-6)
        means_good = False
        weights_good = False
        for p in itertools.permutations([0,1]):
            means_good = means_good or (np.mean(np.abs(sim_m-m[:,p]))<20.0)
            weights_good = weights_good or (np.mean(np.abs(sim_w-w[p,:]))<0.3)
        self.assertTrue(means_good)
        self.assertTrue(weights_good)

    def test_random_means_tsvd_init_m_first(self):
        """
        Test state estimation with random means and weights.

        200 cells, 20 genes, 2 clusters
        """
        sim_m, sim_w = simulation.generate_poisson_states(2, 200, 20)
        sim_data = simulation.generate_state_data(sim_m, sim_w)
        sim_data = sparse.csc_matrix(sim_data)
        m, w, ll = state_estimation.poisson_estimate_state(sim_data, 2, max_iters=10, disp=True, initialization='tsvd', threads=1, run_w_first=False)
        self.assertTrue(np.max(w.sum(0) - 1.0)<0.001)
        obj = sparse_objective(sim_data.data,
                sim_data.indices,
                sim_data.indptr,
                200, 20, m, w)
        self.assertEqual(ll, obj)
        dense_obj = objective(sim_data.toarray(), m, w)
        self.assertTrue(np.abs(obj-dense_obj) < 1e-6)
        means_good = False
        weights_good = False
        for p in itertools.permutations([0,1]):
            means_good = means_good or (np.mean(np.abs(sim_m-m[:,p]))<20.0)
            weights_good = weights_good or (np.mean(np.abs(sim_w-w[p,:]))<0.3)
        self.assertTrue(means_good)
        self.assertTrue(weights_good)

    def test_random_means_tsvd_init_long(self):
        """
        Test state estimation with random means and weights.

        200 cells, 20 genes, 2 clusters
        """
        sim_m, sim_w = simulation.generate_poisson_states(2, 200, 20)
        sim_data = simulation.generate_state_data(sim_m, sim_w)
        sim_data = sparse.csc_matrix(sim_data)
        sim_data.indices = sim_data.indices.astype(np.int64)
        sim_data.indptr = sim_data.indptr.astype(np.int64)
        m, w, ll = state_estimation.poisson_estimate_state(sim_data, 2, max_iters=10, disp=False, initialization='tsvd')
        self.assertTrue(np.max(w.sum(0) - 1.0)<0.001)
        obj = sparse_objective(sim_data.data,
                sim_data.indices,
                sim_data.indptr,
                200, 20, m, w)
        self.assertEqual(ll, obj)
        dense_obj = objective(sim_data.toarray(), m, w)
        self.assertTrue(np.abs(obj-dense_obj) < 1e-6)
        means_good = False
        weights_good = False
        for p in itertools.permutations([0,1]):
            means_good = means_good or (np.mean(np.abs(sim_m-m[:,p]))<20.0)
            weights_good = weights_good or (np.mean(np.abs(sim_w-w[p,:]))<0.3)
        self.assertTrue(means_good)
        self.assertTrue(weights_good)


    def test_random_means_cluster_init(self):
        """
        Test state estimation with random means and weights.

        20 cells, 200 genes, 2 clusters
        """
        sim_m, sim_w = simulation.generate_poisson_states(2, 20, 200)
        sim_data = simulation.generate_state_data(sim_m, sim_w)
        sim_data = sparse.csc_matrix(sim_data)
        m, w, ll = state_estimation.poisson_estimate_state(sim_data, 2, max_iters=10, disp=False, initialization='cluster')
        obj = sparse_objective(sim_data.data,
                sim_data.indices,
                sim_data.indptr,
                20, 200, m, w)
        self.assertEqual(ll, obj)
        dense_obj = objective(sim_data.toarray(), m, w)
        self.assertTrue(np.abs(obj-dense_obj) < 1e-6)
        means_good = False
        weights_good = False
        for p in itertools.permutations([0,1]):
            means_good = means_good or (np.mean(np.abs(sim_m-m[:,p]))<20.0)
            weights_good = weights_good or (np.mean(np.abs(sim_w-w[p,:]))<0.2)
        self.assertTrue(means_good)
        self.assertTrue(weights_good)
