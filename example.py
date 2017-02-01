from scipy.io import loadmat
import numpy as np

import uncurl
from uncurl.pois_ll import poisson_ll_2


if __name__ == '__main__':
    dat = loadmat('data/SCDE_k2_sup.mat')
    data = dat['Dat']
    centers = uncurl.kmeans_pp(data, 3)
    lls = uncurl.poisson_ll(data, centers)
    assignments, centers = uncurl.poisson_cluster(data, 3, init=centers)
