Non-default parameters: things we tried and their results
================================================

There are a number of uncurl parameters (well, not necessarily parameters, more like... run configurations?) that we experimented with. Here's some results.

Cell normalization
------------------

This option involves normalizing the cells by their read counts. First, we calculate the total read count of each cell, and divide all counts for cell i by its total read count. Then, we find the median total read count over all cells, and multiply the entire matrix by that value.

The clustering performance after cell normalization were substantially better on count-valued datasets, and either had no effect or were marginally worse on RPKM-normalized and other forms of data that have already been normalized in some other way. So we would suggest using this option for unnormalized count-valued datasets. The downside is that it might lose some information (if certain cell types were correlated to larger read counts?), but I'm not sure if that happens in practice.

[TODO: include graphs]

To use this option, run ``data_normalized = uncurl.preprocessing.cell_normalize(data)`` and run uncurl on ``data_normalized``.


Constrained W
-------------

When this option is activated, the ``W`` matrix is normalized so that its columns sum to 1 after each round of alternating minimization. Without this option, ``W`` is only constrained to be nonnegative during the optimization process, and normalized after the end of the optimization.

In clustering experiments, this option had mixed results. It performed marginally better on some datasets and marginally worse on others. On the 10X datasets, constrained W performed slightly better when combined with cell normalization, and worse without cell normalization.

[TODO: include graphs]

To use this option, add the argument ``constrain_w=True`` to ``run_state_estimation`` or ``poisson_estimate_state``. This does not work for the NMF-based methods.


Uncurl initialization options
-----------------------------

We provide a variety of initialization options for uncurl. Most initialization methods first perform a clustering, initialize M based on the cluster means, and W based on the cluster assignments. The default initialization is based on truncated SVD followed by K-means. We also provide initializations based on Poisson clustering, and Poisson k-means++ with randomized W. 

In clustering experiments, truncated SVD initialization usually performed the best, but there were some datasets under which Poisson clustering initialization performed better. For example, on randomly downsampled data, Poisson clustering initialization seems to perform better.

To use different initializations, use the argument ``initialization=<method>``, where ``<method>`` can be one of ``tsvd`` (truncated SVD + K-means), ``cluster`` (Poisson clustering), ``kmpp`` (Poisson k-means++), or ``km`` (k-means on the full data).


Alternative to QualNorm: mean-normalized initialization
-------------------------------------------------------

Given prior gene expression data, there are a variety of methods for initializing uncurl. ``QualNorm`` is one way of doing this initialization. Another way, when we have real-valued prior data, we could normalize the prior data so that each cell type sums to 1, and then multiply that by the mean per-cell read count of the actual data.

This performed better than QualNorm on sparse datasets such as the 10X datasets.
