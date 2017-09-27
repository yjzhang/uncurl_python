# downsampling count datasets (for comparisons)

import numpy as np

from scipy import sparse

def downsample(data, percent):
    """
    downsample the data by removing a given percentage of the reads.

    Args:
        data: genes x cells array or sparse matrix
        percent: float between 0 and 1
    """
    n_genes = data.shape[0]
    n_cells = data.shape[1]
    new_data = data.copy()
    total_count = float(data.sum())
    to_remove = total_count*percent
    # sum of read counts per cell
    cell_sums = data.sum(0).astype(float)
    # probability of selecting genes per cell
    cell_gene_probs = data/cell_sums
    # probability of selecting cells
    cell_probs = np.array(cell_sums/total_count).flatten()
    cells_selected = np.random.multinomial(to_remove, pvals=cell_probs)
    for i, num_selected in enumerate(cells_selected):
        cell_gene = np.array(cell_gene_probs[:,i]).flatten()
        genes_selected = np.random.multinomial(num_selected, pvals=cell_gene)
        if sparse.issparse(data):
            genes_selected = sparse.csc_matrix(genes_selected).T
        new_data[:,i] -= genes_selected
    new_data[new_data < 0] = 0
    return new_data
