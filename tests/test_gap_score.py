"""
Using gap score to determine optimal cluster number
"""

import unittest
from unittest import TestCase

import numpy as np
import scipy

from uncurl import gap_score

class GapScoreTest(TestCase):

    def setUp(self):
        pass

    def test_gap_score(self):
        data_mat = scipy.io.loadmat('data/10x_pooled_400.mat')
        data = data_mat['data']
        data_tsvd = gap_score.preproc_data(data)
        max_k, gap_vals, sk_vals = gap_score.run_gap_k_selection(data_tsvd,
                k_min=1, k_max=50, skip=5, B=5)
        # just test that the score is in a very broad range
        self.assertTrue(max_k > 3)
        self.assertTrue(max_k < 20)

    def test_gap_score_2(self):
        data_mat = scipy.io.loadmat('data/GSE60361_dat.mat')
        data = data_mat['Dat']
        data_tsvd = gap_score.preproc_data(data)
        max_k, gap_vals, sk_vals = gap_score.run_gap_k_selection(data_tsvd,
                k_min=1, k_max=50, skip=5, B=5)
        self.assertTrue(max_k > 3)
        self.assertTrue(max_k < 30)


if __name__ == '__main__':
    unittest.main()


