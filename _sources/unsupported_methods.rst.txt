Details on unsupported methods
==============================

There are a number of unsupported or experimental methods part of the UNCURL package. We provide information on them here for the sake of completeness but cannot vouch for their correctness.

Alternative state estimation methods
------------------------------------

We provide implementations of the convex mixture model for the negative binomial (NB) and zero-inflated Poisson (ZIP) distributions. In our experiments they did not work as well as the Poisson model on most datasets.

Alternative clustering methods
------------------------------

As with state estimation, we provide NB and ZIP versions of k-means.

Ensemble Methods
----------------

Consensus clustering, etc. 


Dimensionality reduction
------------------------

The ``mds`` function performs dimensionality reduction using MDS. This works by running MDS on M to convert it into a projection matrix, and then using that matrix to project W onto 2d space. This is much faster than tSNE or even PCA, at the cost of some fidelity, but it might work as a first pass.

Example:

.. code-block:: python

    import numpy as np
    from uncurl import mds, dim_reduce_data

    data = np.loadtxt('counts.txt')

    # dimensionality reduction using MDS on state estimation means
    M, W, ll = poisson_estimate_state(data, 4)
    # proj is a 2d projection of the data.
    proj = mds(M, W, 2)


Lineage estimation
------------------

The ``lineage`` function performs lineage estimation from the output of ``poisson_estimate_state``. It fits the data to a different 5th degree polynomial for each cell type.

The ``pseudotime`` function calculates the pseudotime for each cell given the output of ``lineage`` and a starting cell.

Example (including visualization):

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt

    from uncurl import poisson_estimate_state, mds, lineage, pseudotime

    data = np.loadtxt('counts.txt')

    # pretend that there are three natural clusters in the dataset.
    M, W = poisson_estimate_state(data, 3)

    curve_params, smoothed_points, edges, cell_assignments = lineage(M, W)

    # assume the "root" is cell 0
    ptime = pseudotime(0, edges, smoothed_points)

    # visualizing the lineage
    proj = mds(M, W, 2)

    plt.scatter(proj[0,:], proj[1,:], s=30, c=cell_assignments, edgecolors='none', alpha=0.7)
    plt.scatter(smoothed_points[0,:], smoothed_points[1,:], s=30, c=cell_assignments, edgecolors='none', alpha=0.7)
    # connect the lines
    for edge in edges:
        plt.plot((smoothed_points[0, edge[0]], smoothed_points[0, edge[1]]),
                 (smoothed_points[1, edge[0]], smoothed_points[1, edge[1]]), 'black', linewidth=2)
    plt.xlabel('dim 1')
    plt.ylabel('dim 2')


Visualization
-------------

see ``vis.py``
