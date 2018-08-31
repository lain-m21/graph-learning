from collections import deque
import numpy as np
from numba import jit
from scipy import sparse


@jit('Tuple((i8[:, :], i8[:]))(i8[:, :])')
def matrix_to_edgelist(adj_matrix):
    """
    Now adj_matrix with only integer weights is accepted.

    :param adj_matrix:
    :return:
    """
    edge_list = []
    edge_weights = []
    start_nodes = np.arange(adj_matrix.shape[0])
    for start in start_nodes:
        end_nodes = np.nonzero(adj_matrix[start].reshape(-1, ))[0]
        end_nodes = end_nodes[end_nodes > start]
        for end in end_nodes:
            edge_list.append((start, end))
            edge_weights.append(adj_matrix[start, end])

    edge_list = np.array(edge_list, dtype=int)
    edge_weights = np.array(edge_weights, dtype=int)

    return edge_list, edge_weights


@jit('i8[:, :](i8[:, :])')
def edgelist_to_matrix(edge_list):
    n = np.max(edge_list) + 1
    adj_matrix = np.zeros([n, n], dtype=np.uint8)
    for edge in edge_list:
        adj_matrix[edge[0], edge[1]] = 1
        adj_matrix[edge[1], edge[0]] = 1
    return adj_matrix


def dict_to_matrix(dic):
    n = len(dic.keys())
    adj_matrix = np.zeros([n, n], dtype=np.uint8)
    for key in dic.keys():
        nodes = np.array(list(dic[key]))
        adj_matrix[key, nodes] = 1
    return adj_matrix


@jit('Tuple((i8[:, :], i8[:, :]))(i8[:, :], f8, b1)')
def split_graph(edge_list, test_ratio=0.2, avoid_alone=True):
    node_degree = np.bincount(np.sort(edge_list.ravel()))
    nodes = np.unique(edge_list)
    split_train = deque([])
    split_test = deque([])
    for n in nodes:
        edges = edge_list[edge_list[:, 0] == n]
        num_split_test = int(node_degree[n] * test_ratio)
        num_split_train = node_degree[n] - num_split_test
        if num_split_train < 1:
            split_test.append(edges)
            split_train.append(edges)
            continue

        candidates = deque([])
        for edge in edges:
            if node_degree[edge[1]] > 2:
                candidates.append(edge)
            else:
                split_train.append(edge.reshape(1, 2))
                split_test.append(edge.reshape(1, 2))
                
        candidates = np.array(candidates)
        if len(candidates) < 1:
            continue
        split_test_ends = np.random.choice(candidates[:, 1], num_split_test)
        split_train_ends = np.setdiff1d(edges[:, 1], split_test_ends)

        split_test_edges = np.ones([len(split_test_ends), 2], dtype=int) * n
        split_train_edges = np.ones([len(split_train_ends), 2], dtype=int) * n

        split_test_edges[:, 1] = split_test_ends
        split_train_edges[:, 1] = split_train_ends
        split_test.append(split_test_edges)
        split_train.append(split_train_edges)

        for end in split_test_ends:
            node_degree[end] -= 1

    return np.concatenate(split_train, axis=0), np.concatenate(split_test, axis=0)


@jit('i8[:, :](i8[:, :], f8)')
def sample_negative_links(edge_list, neg_ratio=1.0):
    nodes = np.unique(edge_list)
    index_shuffle = np.random.permutation(np.arange(len(nodes)))
    nodes = nodes[index_shuffle]

    neg_size = int(len(edge_list) * neg_ratio)
    result = deque([])
    for i, start in enumerate(nodes):
        end_nodes = nodes[nodes > start]
        drop_nodes = edge_list[edge_list[:, 0] == start][:, 1]
        end_nodes = np.setdiff1d(end_nodes, drop_nodes)
        for end in end_nodes:
            edge = np.array([start, end], dtype=int)
            result.append(edge)
            if len(result) == neg_size:
                break
        if len(result) == neg_size:
            break
    return np.array(result)
