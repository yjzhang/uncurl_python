from unittest import TestCase

import numpy as np

from uncurl import simulation, lineage

class LineageTest(TestCase):

    def setUp(self):
        pass

    def test_lineage(self):
        """
        Testing lineage using provided weights and means
        """
        M, W = simulation.generate_poisson_lineage(3, 100, 50)
        sim_data = simulation.generate_state_data(M, W)
        sim_data = sim_data + 1e-8
        m2 = M + np.random.random(M.shape) - 0.5
        curves, fitted_vals, edges, assignments = lineage(sim_data, m2, W)
        # TODO: assert something about the distances???
        print len(edges)
        adjacent_count = 0
        for e in edges:
            if np.abs(e[0]-e[1]) <= 1:
                adjacent_count += 1
        self.assertTrue(adjacent_count>150)
