import numpy as np
from numba import jit


@jit('f8[:](i8[:, :], i8[:, :])')
def jaccard_coefficient(adj_matrix, edge_list):
    """
    Compute Jaccard coefficient for each edge in edge_list using adjacency matrix.
    Please use this for baseline model for link prediction tasks.

    :param adj_matrix: N by N array
    :param edge_list: L by 2 array
    :return:
    """
    results = np.zeros(len(edge_list))
    for i, edge in enumerate(edge_list):
        neighbors_u = adj_matrix[edge[0], :]
        neighbors_v = adj_matrix[edge[1], :]
        num_common_neighbors = np.sum(np.logical_and(neighbors_u, neighbors_v))
        num_union_neighbors = np.sum(np.logical_or(neighbors_u, neighbors_v))
        if num_union_neighbors == 0:
            results[i] = 0
        else:
            jc = num_common_neighbors / num_union_neighbors
            results[i] = jc

    return np.array(results)
