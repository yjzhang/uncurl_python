"""
Using gap score to determine optimal cluster number
"""

import unittest
from unittest import TestCase
from flaky import flaky

import numpy as np
import scipy

from uncurl import gap_score

class GapScoreTest(TestCase):

    def setUp(self):
        pass

    def test_gap_score(self):
        data_mat = scipy.io.loadmat('data/10x_pooled_400.mat')
        data = data_mat['data']
        data_tsvd = gap_score.preproc_data(data, gene_subset=True)
        max_k, gap_vals, sk_vals = gap_score.run_gap_k_selection(data_tsvd,
                k_min=1, k_max=50, skip=5, B=5)
        # just test that the score is in a very broad range
        self.assertTrue(max_k > 3)
        self.assertTrue(max_k < 20)

    def test_gap_score_2(self):
        data_mat = scipy.io.loadmat('data/GSE60361_dat.mat')
        data = data_mat['Dat']
        data_tsvd = gap_score.preproc_data(data, gene_subset=True)
        max_k, gap_vals, sk_vals = gap_score.run_gap_k_selection(data_tsvd,
                k_min=1, k_max=50, skip=5, B=5)
        self.assertTrue(max_k > 3)
        self.assertTrue(max_k < 30)

    @flaky(max_runs=3)
    def test_gap_score_3(self):
        data_mat = scipy.io.loadmat('data/SCDE_test.mat')
        data = data_mat['dat']
        data_tsvd = gap_score.preproc_data(data, gene_subset=True)
        max_k, gap_vals, sk_vals = gap_score.run_gap_k_selection(data_tsvd,
                k_min=1, k_max=50, skip=5, B=5)
        self.assertTrue(max_k < 10)



if __name__ == '__main__':
    unittest.main()


