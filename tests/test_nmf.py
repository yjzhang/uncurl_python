from __future__ import print_function

from unittest import TestCase

import numpy as np
from scipy import sparse
from scipy.io import loadmat

import uncurl

class NMFTest(TestCase):

    def setUp(self):
        dat = loadmat('data/SCDE_test.mat')
        self.data = dat['dat'].toarray()[0:500, :]
        self.data_sparse = sparse.csc_matrix(self.data)
        self.labs = dat['Lab'][0]

    def test_run_lognorm_nmf(self):
        w, h, cost = uncurl.nmf_wrapper.log_norm_nmf(self.data, 2)
        labs = h.argmax(0)
        self.assertTrue(uncurl.evaluation.purity(labs, self.labs) > 0.85)

    def test_run_norm_nmf(self):
        w, h, cost = uncurl.nmf_wrapper.norm_nmf(self.data, 2)
        labs = h.argmax(0)
        self.assertTrue(uncurl.evaluation.purity(labs, self.labs) > 0.8)

    def test_run_se(self):
        w, h, cost = uncurl.run_state_estimation(self.data, 2, dist='log-norm')
        labs = h.argmax(0)
        self.assertTrue(uncurl.evaluation.purity(labs, self.labs) > 0.85)
        w1, h1, cost = uncurl.run_state_estimation(self.data, 2, dist='gaussian')
        labs = h1.argmax(0)
        self.assertTrue(uncurl.evaluation.purity(labs, self.labs) > 0.8)
