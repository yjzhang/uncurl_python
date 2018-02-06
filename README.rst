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

The simplest way to use state estimation is to use the ``run_state_estimation`` function, which can be used to call any of the state estimation functions for different distributions. The possible distributions are 'Poiss', 'NB', 'ZIP', or 'LogNorm'. Generally, 'Poiss' is recommended for sparse or count-valued datasets. Currently the NB and ZIP options are unsupported.

Before running state estimation, it is often a good idea to subset the number of genes. This can be done using the function ``max_variance_genes``, which bins the genes by mean expression, and selects a top fraction of genes by variance from each bin. It also removes genes that have all zero expression counts.

Example:

.. code-block:: python

    import numpy as np
    import scipy.io
    from uncurl import max_variance_genes, run_state_estimation

    data = np.loadtxt('counts.txt')

    # sparse data (matrix market format)
    data_sparse = scipy.io.mmread('matrix.mtx')

    # max variance genes, default parameters 
    genes = max_variance_genes(data_sparse, nbins=5, frac=0.2)
    data_subset = data_sparse[genes,:]

    M, W, ll = run_state_estimation(data_subset, clusters=4, dist='Poiss', disp=False, max_iters=30, inner_max_iters=100, initialization='tsvd', threads=8)

    M2, W2, cost = run_state_estimation(data_subset, clusters=4, dist='LogNorm')

Details
^^^^^^^
``run_state_estimation`` is actually a wrapper around several other functions for state estimation.

The ``poisson_estimate_state`` function is used to estimate cell types using the Poisson Convex Mixture Model. It can take in dense or sparse matrices of reals or integers as input, and can be accelerated by parallelization. The input is of shape (genes, cells). It has three outputs: two matrices ``M`` and ``W``, and ``ll``, the negative log-likelihood. M is a (genes, clusters) matrix, and W is a (clusters, cells) matrix where each column sums to 1. The outputs ``W`` and ``M*W`` can be used for further visualization or dimensionality reduction, as described latter.

There are a number of different initialization methods and options for ``poisson_estimate_state``. By default, it is initialized using truncated SVD + K-means, but it can also be initialized using ``poisson_cluster`` or just K-means.

Example:

.. code-block:: python

    from uncurl import max_variance_genes, poisson_cluster, poisson_estimate_state

    # poisson state estimation
    M, W, ll = poisson_estimate_state(data_subset, 2)

    # labels in 0...k-1
    labels = W.argmax(0)

    # optional arguments
    M, W, ll = poisson_estimate_state(data_subset, clusters=2, disp=False, max_iters=30, inner_max_iters=150, initialization='tsvd', threads=8)

    # initialization by providing means and weights
    assignments_p, centers = poisson_cluster(data_subset, 2)
    M, W, ll = poisson_estimate_state(data_subset, 2, init_means=centers, init_weights=assignments_p)

The ``log_norm_nmf`` function is a wrapper around scikit-Learn's NMF class that performs a log-transform and per-cell count normalization before running NMF. It returns two matrices, W and H, which correspond to the M and W returned by ``poisson_estimate_state``. It can also take sparse matrix inputs.

Example:

.. code-block:: python

    from uncurl import log_norm_nmf

    W, H = log_norm_nmf(data_subset, k=2)


Distribution Selection
----------------------

The ``DistFitDataset`` function is used to determine the distribution of each gene in a dataset by calculating the fit error for the Poisson, Normal, and Log-Normal distributions. It currently only works for dense matrices.

Example:

.. code-block:: python

    import numpy as np
    from uncurl import DistFitDataset

    data = np.loadtxt('counts.txt')

    fit_errors = DistFitDataset(data)

    poiss_fit_errors = fit_errors['poiss']
    norm_fit_errors = fit_errors['norm']
    lognorm_fit_errors = fit_errors['lognorm']


The output, ``fit_errors``, contains the fit error for each gene, for each of the three distributions when fitted to the data using maximum likelihood.


Qualitative to Quantitative Framework
-------------------------------------

The ``qualNorm`` function is used to convert binary (or otherwise) data with shape (genes, types) into starting points for clustering and state estimation.

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

We recommend using standard dimensionality reduction techniques such as t-SNE and PCA. They can be run on either W or ``MW = M.dot(W)``. When running t-SNE on MW, we suggest taking the log and then doing a PCA or truncated SVD, as you would do for the original input data. This is the basis for the UNCURL + tSNE results in our paper. When using t-SNE on W, we suggest using a symmetric relative entropy metric, which is available as ``uncurl.sparse_utils.symmetric_kld`` (this can be passed in to scikit-learn's t-SNE implementation). Cosine distance has also worked better than Euclidean distance on W.

Alternatively, we provide an MDS-based dimensionality reduction method that takes advantage of the convex mixture model. It is generally less accurate than t-SNE, but much faster. See `docs for unsupported methods <https://yjzhang.github.io/uncurl_python/unsupported_methods.html#dimensionality-reduction>`_.


Lineage Estimation & Pseudotime
-------------------------------

The output MW of UNCURL can be used as input for other lineage estimation tools.

We also have implemented our own lineage estimation tools but have not thoroughly validated them. See `docs for unsupported methods <https://yjzhang.github.io/uncurl_python/unsupported_methods.html#lineage-estimation>`_.


Included datasets
-----------------

Real datasets:

10x_pooled_400.mat: 50 cells each from 8 cell types: CD19+ b cells, CD14+ monocytes, CD34+, CD56+ NK, CD4+/CD45RO+ memory t, CD8+/CD45RA+ naive cytotoxic, CD4+/CD45RA+/CD25- naive t, and CD4+/CD25 regulatory t. Source: `10x genomics <https://support.10xgenomics.com/single-cell-gene-expression/datasets>`_.

GSE60361_dat.mat: subset of data from `Zelsel et al. 2015 <http://linnarssonlab.org/cortex>`_.

SCDE_test.mat: data from `Islam et al. 2011 <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE29087>`_.

Synthetic datasets:

BranchedSynDat.mat: simulated lineage dataset with 3 branches

SynMouseESprog_1000.mat: simulated lineage dataset showing linear differentiation
