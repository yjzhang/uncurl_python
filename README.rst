UNCURL
======

To install after cloning the repo: ``pip install .``

Requirements: numpy, scipy, cython, scikit-learn

To run tests: ``python setup.py test``

Examples: see the examples folder.

`Full documentation <https://yjzhang.github.io/uncurl_python/>`_

Features
========

Clustering
----------

The ``poisson_cluster`` function does Poisson clustering with hard assignments. It takes an array of features by examples and the number of clusters, and returns two arrays: an array of cluster assignments and an array of cluster centers.

The ``nb_cluster`` function is used for negative binomial clustering with the same parameters. It returns three arrays: P and R, the negative binomial parameters for all genes and clusters, and the cluster assignments for each cell.

Example:

.. code-block:: python

  from uncurl import poisson_cluster, nb_cluster
  import numpy as np

  # data is a 2d array of floats, with dimensions genes x cells
  data = np.loadtxt('counts.txt')
  assignments_p, centers = poisson_cluster(data, 2)
  assignments_nb, P, R = nb_cluster(data, 2)


Qualitative to Quantitative Framework
-------------------------------------

The ``qualNorm`` function is used to convert binary data into starting points for clustering.

Example:

.. code-block:: python

    from uncurl import qualNorm
    import numpy as np

    data = np.loadtxt('counts.txt')
    bin_data = np.loadtxt('binary.txt')
    starting_centers = qualNorm(data, bin_data)
    assignments, centers = poisson_cluster(data, 2, init=starting_centers)

State Estimation
----------------

The ``poisson_estimate_state`` function is used to estimate cell types using the Poisson Convex Mixture Model. It can take in dense or sparse matrices of reals or integers as input, and can be accelerated by parallelization. 

The ``nb_estimate_state`` function has a similar output, but uses a negative binomial distribution, is much slower, and is not capable of dealing with sparse inputs.

There are a number of different initialization methods and options for ``poisson_estimate_state``. By default, it is initialized using ``poisson_cluster``, but it can also be initialized using truncated SVD + K-means or just K-means.

Example:

.. code-block:: python

    from uncurl import poisson_estimate_state, nb_estimate_state

    data = np.loadtxt('counts.txt')

    M, W, ll = poisson_estimate_state(data, 2)
    M2, W2, R, ll2 = nb_estimate_state(data, 2)

    # optional arguments
    M, W, ll = poisson_estimate_state(data, 2, disp=False, max_iters=10, inner_max_iters=200, initialization='tsvd', threads=8)

    # initialization by providing means and weights
    assignments_p, centers = poisson_cluster(data, 2)
    M, W, ll = poisson_estimate_state(data, 2, init_means=centers, init_weights=assignments_p)


Dimensionality Reduction
------------------------

The ``dim_reduce_data`` function performs dimensionality reduction using MDS. Alternatively, dimensionality reduction can be performed using the results of state estimation, by converting the output means of state estimation into a projection matrix. 

Example:

.. code-block:: python

    import numpy as np
    from uncurl import dim_reduce, dim_reduce_data

    data = np.loadtxt('counts.txt')
    data_reduced = dim_reduce_data(data, 2)

    # dimensionality reduction using MDS on state estimation means
    M, W, ll = poisson_estimate_state(data, 2)
    X = dim_reduce(M, W, 2)
    # proj is a 2d projection of the data.
    proj = np.dot(X, W)


Lineage Estimation & Pseudotime
-------------------------------

The ``lineage`` function performs lineage estimation from the output of ``poisson_estimate_state``. It fits the data to a different 5th degree polynomial for each cell type.

The ``pseudotime`` function calculates the pseudotime for each cell given the output of ``lineage`` and a starting cell.

Example (including visualization):

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt

    from uncurl import poisson_estimate_state, dim_reduce_data, lineage, pseudotime

    data = np.loadtxt('counts.txt')
    # pretend that there are three natural clusters in the dataset.
    M, W = poisson_estimate_state(data, 3)

    curve_params, smoothed_points, edges, cell_assignments = lineage(M, W)

    # assume the "root" is cell 0
    ptime = pseudotime(0, edges, smoothed_points)

    # visualizing the lineage
    X = dim_reduce_data(M, 2)
    proj = np.dot(X.T, W)

    plt.scatter(proj[0,:], proj[1,:], s=30, c=cell_assignments, edgecolors='none', alpha=0.7)
    plt.scatter(smoothed_points[0,:], smoothed_points[1,:], s=30, c=cell_assignments, edgecolors='none', alpha=0.7)
    # connect the lines
    for edge in edges:
        plt.plot((smoothed_points[0, edge[0]], smoothed_points[0, edge[1]]),
                (smoothed_points[1, edge[0]], smoothed_points[1, edge[1]]), 'black', linewidth=2)
    plt.xlabel('dim 1')
    plt.ylabel('dim 2')
