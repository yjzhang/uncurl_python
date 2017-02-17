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
        M, W = simulation.generate_poisson_lineage(3, 100, 200)
        sim_data = simulation.generate_state_data(M, W)
        sim_data = sim_data + 1e-8
        m2 = M + np.random.random(M.shape) - 0.5
        # TODO: need a better simulation
        curves, fitted_vals, edges, assignments = lineage(sim_data, m2, W)
        # assert something about the distances???
        # 1-NN based error?
