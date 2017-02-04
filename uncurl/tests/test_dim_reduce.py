from unittest import TestCase

import numpy as np

from uncurl import simulation, dim_reduce

class DimReduceTest(TestCase):

    def setUp(self):
        pass

    def test_dim_reduce(self):
        """
        Test dimensionality reduction using sample data
        """
        sim_means = np.array([[20.,30.,1.],
                              [10.,3.,8.],
                              [90.,50.,20.],
                              [10.,4.,30.]])
        sim_assignments = np.array([[0.1,0.2,0.3,0.4,0.5,0.1,0.8],
                                    [0.5,0.3,0.2,0.4,0.2,0.2,0.1],
                                    [0.4,0.5,0.5,0.2,0.3,0.7,0.1]])
        sim_data = simulation.generate_state_data(sim_means, sim_assignments)
        sim_data = sim_data + 1e-8
        X = dim_reduce(sim_data, sim_means, sim_assignments, 2)
        self.assertEqual(X.shape, (3, 2))
        projections = np.dot(X.transpose(), sim_assignments)
        # assert something about the distances???
        # 1-NN based error?
