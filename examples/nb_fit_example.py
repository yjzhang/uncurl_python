import numpy as np
import matplotlib.pyplot as plt

from uncurl import nb_cluster, simulation
from uncurl.nb_cluster import nb_ll, nb_fit

if __name__=='__main__':
    P = np.array([[0.5],
                [0.9],
                [0.4]])
    R = np.array([[1.],
                [5.],
                [2.]])
    data = simulation.generate_nb_data(P, R, 100)
    print data
    p, r = nb_fit(data)
    print p
    print r
    sim_data = simulation.generate_nb_data(p.reshape((3,1)), r.reshape((3,1)), 100)
    plt.hist(data[1], alpha=0.5)
    plt.hist(sim_data[1], alpha=0.5)
    plt.show()

