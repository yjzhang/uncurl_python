# -*- coding: utf-8 -*-
"""
Created on Thu Nov 09 09:13:05 2017

@author: Sumit
"""

import scipy.io as scio
import numpy as np 
import Cython
import pyximport 
pyximport.install(setup_args={"include_dirs":np.get_include()})
import state_estimation_tsvd as se
import preprocessing as prep
from scipy import sparse
from preprocessing import cell_normalize
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi


Dat = scio.loadmat('E:/UWPhDWork/SCS/DropBarcoding/CellClassificationData/GSE75413_dat')
Dat2 = np.log(1 + cell_normalize(Dat['Dat']))
Labs = Dat['Lab']

In = prep.max_variance_genes(Dat2)

Dat2 = Dat2[In]

M, W, obj = se.poisson_se_multiclust(Dat2,3)
PredLab = np.argmax(W, axis = 0)
Labs = np.reshape(Labs, len(Labs.T))
print nmi(Labs, PredLab)