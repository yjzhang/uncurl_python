UNCURL
======

To install after cloning the repo: `pip install .`

To run tests: `python setup.py test`

Examples: see `example.py`

## Features

### Clustering

The `poisson_cluster` function does Poisson clustering with hard assignments. It takes an array of features by examples and the number of clusters, and returns two arrays: an array of cluster assignments and an array of cluster centers.

Example:

```python
from uncurl import poisson_cluster
import numpy as np

data = np.loadtxt('counts.txt')
assignments, centers = poisson_cluster(data, 2)
```

### Qualitative to Quantitative Framework

### State Estimation

