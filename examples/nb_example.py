from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

import uncurl
from uncurl import nb_cluster, simulation

if __name__ == '__main__':
    P = np.array([[0.5],
                  [0.8],
                  [0.1]])
    R = np.array([[10.],
                  [30.],
                  [20.]])
    data = simulation.generate_nb_data(P, R, 10) 
    print data
    print np.mean(data, 1)
    print np.var(data, 1)
    P_new, R_new = nb_cluster.nb_fit(data, max_iters=500)
    print P_new, R_new
    data2 = simulation.generate_nb_data(P_new.reshape((3,1)), R_new.reshape((3,1)), 50) 
    P_new1, R_new1 = nb_cluster.nb_fit(data, max_iters=500)
    print P_new1, R_new1
    print data2
    print np.mean(data2, 1)
    print np.var(data2, 1)
    P = np.array([[0.5,0.8],
                  [0.8,0.1],
                  [0.1,0.5]])
    R = np.array([[10.,300.],
                  [30.,200.],
                  [20.,100.]])
    assignments = [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]
    data = simulation.generate_nb_data(P, R, len(assignments), assignments)
    P_, R_, assignments = nb_cluster.nb_cluster(data, 2)
    print 'clustering results'
    print P_
    print R_
    print assignments
