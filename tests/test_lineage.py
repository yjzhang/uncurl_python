from unittest import TestCase
from flaky import flaky

import numpy as np

from uncurl import simulation, run_lineage, pseudotime

@flaky
class LineageTest(TestCase):

    def setUp(self):
        pass

    def test_lineage(self):
        """
        Testing lineage using randomly generated lineage data
        """
        M, W = simulation.generate_poisson_lineage(3, 100, 50)
        sim_data = simulation.generate_state_data(M, W)
        sim_data = sim_data + 1e-8
        m2 = M + np.random.random(M.shape) - 0.5
        curves, fitted_vals, edges, assignments = run_lineage(m2, W)
        # TODO: assert something about the distances???
        print(len(edges))
        adjacent_count = 0
        for e in edges:
            if np.abs(e[0]-e[1]) <= 1:
                adjacent_count += 1
        self.assertTrue(adjacent_count>150)

    def test_pseudotime(self):
        """
        Test pseudotime calculations
        """
        M, W = simulation.generate_poisson_lineage(3, 100, 50)
        sim_data = simulation.generate_state_data(M, W)
        sim_data = sim_data + 1e-8
        m2 = M + np.random.random(M.shape) - 0.5
        curves, fitted_vals, edges, assignments = run_lineage(m2, W)
        ptime = pseudotime(0, edges, fitted_vals)
        # assert that the cells are generally increasing in ptime
        # test each cluster
        old_p = 0
        for i in range(100):
            p = ptime[i]
            self.assertTrue(p >= old_p)
            old_p = p
        old_p = 0
        for i in range(100, 200):
            p = ptime[i]
            self.assertTrue(p >= old_p)
            self.assertTrue(p > 0)
            old_p = p
        old_p = 0
        for i in range(200, 300):
            p = ptime[i]
            self.assertTrue(p >= old_p)
            self.assertTrue(p > 0)
            old_p = p
