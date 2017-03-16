UNCURL
======

TODO: pypi

To install after cloning the repo: `pip install .`

To run tests: `python setup.py test`

Examples: see the examples folder.

## Features

### Clustering

The `poisson_cluster` function does Poisson clustering with hard assignments. It takes an array of features by examples and the number of clusters, and returns two arrays: an array of cluster assignments and an array of cluster centers.

The `nb_cluster` function is used for negative binomial clustering with the same parameters. It returns three arrays: P and R, the negative binomial parameters for all genes and clusters, and the cluster assignments for each cell.

Example:

```python
from uncurl import poisson_cluster, nb_cluster
import numpy as np

# data is a 2d array of floats, with dimensions genes x cells
data = np.loadtxt('counts.txt')
assignments_p, centers = poisson_cluster(data, 2)
P, R, assignments_nb = nb_cluster(data, 2)
```


### Qualitative to Quantitative Framework

The `qual2quant` function is used to convert binary data into starting points for clustering.

Example:

```python
from uncurl import qual2quant
import numpy as np

data = np.loadtxt('counts.txt')
bin_data = np.loadtxt('binary.txt')
starting_centers = qual2quant(data, bin_data)
```

### State Estimation

The `poisson_estimate_state` function is used to estimate cell types using the Poisson Convex Mixture Model.

Example:

```python
from uncurl import poisson_estimate_state

data = np.loadtxt('counts.txt')
M, W = poisson_estimate_state(data, 2)
```

### Dimensionality Reduction

The `dim_reduce_data` function performs dimensionality reduction using MDS.

Example:
```python
from uncurl import dim_reduce_data

data = np.loadtxt('counts.txt')
X = dim_reduce_data(data, 2)
```

### Lineage Estimation

The `lineage` function performs lineage estimation from the output of `poisson_estimate_state`. It fits the data to a different 5th degree polynomial for each cell type.

Example (including visualization):

```python
import numpy as np
import matplotlib.pyplot as plt

from uncurl import poisson_estimate_state, dim_reduce_data, lineage

data = np.loadtxt('counts.txt')
# pretend that there are three natural clusters in the dataset.
M, W = poisson_estimate_state(data, 3)

curve_params, smoothed_points, edges, cell_assignments = lineage(M, W)

# visualizing the lineage
X = dim_reduce_data(M, 2)
proj = np.dot(X.T, W)

plt.scatter(proj[0,:], proj[1,:], s=30, c=true_weights.argmax(0), edgecolors='none', alpha=0.7)
plt.scatter(smoothed_points[0,:], smoothed_points[1,:], s=30, c=W.argmax(0), edgecolors='none', alpha=0.7)
# connect the lines
for edge in edges:
    plt.plot((smoothed_points[0, edge[0]], smoothed_points[0, edge[1]]),
            (smoothed_points[1, edge[0]], smoothed_points[1, edge[1]]), 'black', linewidth=2)
plt.xlabel('dim 1')
plt.ylabel('dim 2')
```
