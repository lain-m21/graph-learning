import numpy as np
from numba import jit


@jit('i8[:, :](i8[:, :], i8[:], i8)')
def uniform_neighbor_sampling(adj_list, nodes, num_samples):
    neighbors = adj_list[nodes]
    index_shuffle = np.random.permutation(adj_list.shape[1])
    neighbors = neighbors[:, index_shuffle]
    return neighbors[:, :num_samples]


def convolution_sampling(adj_list, nodes, layer_infos):
    samples = [nodes.reshape(-1, 1)]
    for i, layer in enumerate(layer_infos):
        num_samples = layer['num_samples']
        neighbors = uniform_neighbor_sampling(adj_list, samples[i], num_samples)
        samples.append(neighbors)
    return np.concatenate(samples, axis=1)


@jit('i8[:](i8[:, :], i8[:], f8)')
def prepare_sampling_distribution(adj_matrix, nodes, smoothing=0.75):
    sampling_distribution = np.array(np.power(adj_matrix.sum(axis=1), smoothing)).reshape(-1,)
    sampling_distribution = sampling_distribution / sampling_distribution.sum()
    indices = np.argsort(sampling_distribution)

    sampling_array = np.zeros(10 ** 8, dtype=int)
    start = 0
    for i in indices:
        prob = sampling_distribution[i]
        num_fill = int(round(prob * (10 ** 8)))
        sampling_array[start:start+num_fill] = nodes[i]
        start += num_fill

    return sampling_array


@jit('i8[:](i8[:], f8[:], i8)')
def negative_edge_sampling(nodes_to_exclude, sampling_array, num_samples):
    excludes = set(nodes_to_exclude.tolist())
    negative_samples = []
    while len(negative_samples) < num_samples:
        sample_index = round(np.random.rand() * (10 ** 8))
        sample = sampling_array[sample_index]
        if sample in excludes:
            continue
        negative_samples.append(sample)

    return np.array(negative_samples, dtype=int)


@jit('i8[:, :](i8[:, :], i8[:], i8, i8)')
def random_walk_sampling(adj_matrix, nodes, num_walks=50, walk_length=5):
    pairs = []
    for count, node in enumerate(nodes):
        for i in range(num_walks):
            curr_node = node
            for j in range(walk_length):
                if curr_node != node:
                    pairs.append([node, curr_node])
                neighbors = np.nonzero(adj_matrix[curr_node])[0]
                next_node = np.random.choice(neighbors)
                curr_node = next_node
    return np.array(pairs)


@jit('Tuple((i8[:, :], i8[:]))(i8[:, :], i8[:], i8)')
def construct_adjlist(adj_matrix, nodes, max_degree):
    adj_list = len(nodes) * np.ones([len(nodes), max_degree], dtype=int)

    for node in nodes:
        neighbors = adj_matrix[node, :]
        if len(neighbors) > max_degree:
            neighbors = np.random.choice(neighbors, max_degree, replace=False)
        elif len(neighbors) < max_degree:
            neighbors = np.random.choice(neighbors, max_degree, replace=True)

        adj_list[node, :] = neighbors

    return adj_list
