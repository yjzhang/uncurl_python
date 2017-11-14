UNCURL
======

.. image:: https://travis-ci.org/yjzhang/uncurl_python.svg
    :target: https://travis-ci.org/yjzhang/uncurl_python

After cloning the repository, first run ``pip install -r requirements.txt`` to install the required libraries. Then, run ``pip install .``

Requirements: numpy, scipy, cython, scikit-learn

Tested on python 2.7, 3.4

For parallel state estimation, OpenMP is required.

To run tests: ``python setup.py test``

Examples: see the examples folder.

`Full documentation <https://yjzhang.github.io/uncurl_python/>`_

Features
========

State Estimation
----------------

The ``poisson_estimate_state`` function is used to estimate cell types using the Poisson Convex Mixture Model. It can take in dense or sparse matrices of reals or integers as input, and can be accelerated by parallelization. The input is of shape (genes, cells). It has three outputs: two matrices ``M`` and ``W``, and ``ll``, the negative log-likelihood. M is a (genes, clusters) matrix, and W is a (clusters, cells) matrix where each column sums to 1. The outputs ``W`` and ``M*W`` can be used for further visualization or dimensionality reduction, such as with t-SNE, or the MDS-based method described later.

Before running state estimation, it is often a good idea to subset the number of genes. This can be done using the function ``max_variance_genes``, which bins the genes by mean expression, and selects a top fraction of genes by variance from each bin. It also removes genes that have all zero expression counts.

The ``log_norm_nmf`` function is a wrapper around scikit-Learn's NMF class that performs a log-transform and per-cell count normalization before running NMF. It returns two matrices, W and H, which correspond to the M and W returned by ``poisson_estimate_state``.

There are a number of different initialization methods and options for ``poisson_estimate_state``. By default, it is initialized using ``poisson_cluster``, but it can also be initialized using truncated SVD + K-means or just K-means.

Example:

.. code-block:: python

    import numpy as np
    import scipy.io
    from uncurl import max_variance_genes, poisson_cluster, poisson_estimate_state

    data = np.loadtxt('counts.txt')

    # sparse data (matrix market format)
    data_sparse = scipy.io.mmread('matrix.mtx')

    # max variance genes, default parameters
    genes = max_variance_genes(data_sparse, nbins=5, frac=0.2)
    data_subset = data_sparse[genes,:]

    # poisson state estimation
    M, W, ll = poisson_estimate_state(data_subset, 2)

    # labels in 0...k-1
    labels = W.argmax(0)

    # optional arguments
    M, W, ll = poisson_estimate_state(data_subset, clusters=2, disp=False, max_iters=30, inner_max_iters=150, initialization='tsvd', threads=8)

    # initialization by providing means and weights
    assignments_p, centers = poisson_cluster(data_subset, 2)
    M, W, ll = poisson_estimate_state(data_subset, 2, init_means=centers, init_weights=assignments_p)


Distribution Selection
----------------------

The ``GetDistFitError`` function is used to determine the distribution of each gene in a dataset by calculating the fit error for the Poisson, Normal, and Log-Normal distributions. It currently only works for dense matrices.

Example:

.. code-block:: python

    import numpy as np
    from uncurl import GetDistFitError

    data = np.loadtxt('counts.txt')

    fit_errors = GetDistFitError(data)

    poiss_fit_error = fit_errors['poiss']
    norm_fit_error = fit_errors['norm']
    lognorm_fit_errors = fit_errors['lognorm']



Qualitative to Quantitative Framework
-------------------------------------

The ``qualNorm`` function is used to convert binary data with shape (genes, types) into starting points for clustering and state estimation.

Example:

.. code-block:: python

    from uncurl import qualNorm
    import numpy as np

    data = np.loadtxt('counts.txt')
    bin_data = np.loadtxt('binary.txt')
    starting_centers = qualNorm(data, bin_data)
    assignments, centers = poisson_cluster(data, 2, init=starting_centers)


Clustering
----------

The ``poisson_cluster`` function does Poisson clustering with hard assignments. It takes an array of features by examples and the number of clusters, and returns two arrays: an array of cluster assignments and an array of cluster centers.


Example:

.. code-block:: python

  from uncurl import poisson_cluster
  import numpy as np

  # data is a 2d array of floats, with dimensions genes x cells
  data = np.loadtxt('counts.txt')
  assignments_p, centers = poisson_cluster(data, 2)


Dimensionality Reduction
------------------------

Dimensionality reduction can be performed using the results of state estimation, by converting the output means of state estimation into a projection matrix. 

Alternatively, ``dim_reduce_data`` function performs dimensionality reduction using MDS. 

Example:

.. code-block:: python

    import numpy as np
    from uncurl import mds, dim_reduce_data

    data = np.loadtxt('counts.txt')

    # dimensionality reduction using MDS on state estimation means
    M, W, ll = poisson_estimate_state(data, 2)
    # proj is a 2d projection of the data.
    proj = mds(M, W, 2)

    # you should probably use mds from scikit-learn instead of this method.
    data_reduced = dim_reduce_data(data, 2)


In addition to using MDS, it's easy to use standard dimensionality reduction techniques such as t-SNE and PCA. When using t-SNE on W (from ``poisson_estimate_state``), we recommend using a symmetric relative entropy based metric, which is available as ``uncurl.sparse_utils.symmetric_kld``. Cosine distance has also worked better than Euclidean distance on W.


Lineage Estimation & Pseudotime
-------------------------------

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
