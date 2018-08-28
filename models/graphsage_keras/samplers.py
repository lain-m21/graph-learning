import numpy as np
from numba import jit


@jit
def uniform_neighbor_sampling(nodes, adj_list, num_samples):
    neighbors = adj_list[nodes]
    index_shuffle = np.random.permutation(adj_list.shape[1])
    neighbors = neighbors[:, index_shuffle]
    return neighbors[:, :num_samples]