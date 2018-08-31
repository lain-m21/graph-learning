import numpy as np
from numba import jit
from keras.utils import Sequence

from .samplers import prepare_sampling_distribution, negative_edge_sampling
from .samplers import random_walk_sampling, construct_adjlist, convolution_sampling


class UnsupervisedEdgeGenerator(Sequence):
    def __init__(self, adj_matrix, edge_list, layer_infos, max_degree=10, num_neg_samples=20,
                 num_walks=50, walk_length=5, batch_size=256, shuffle=True):
        self.nodes = np.unique(edge_list)
        # self.edges = random_walk_sampling(adj_matrix, self.nodes, num_walks, walk_length)
        self.edges = edge_list
        self.adj_matrix = adj_matrix
        self.layer_infos = layer_infos
        self.num_walks = num_walks
        self.walk_length = walk_length

        self.max_degree = max_degree
        self.batch_size = batch_size
        self.num_neg_samples = num_neg_samples
        self.shuffle = shuffle

        self.num_data = len(self.edges)
        self.indexes = np.arange(self.num_data)

        self.sampling_array = prepare_sampling_distribution(self.adj_matrix, self.nodes, smoothing=0.75)
        self.adj_list = construct_adjlist(self.adj_matrix, self.nodes, self.max_degree)

    def on_epoch_end(self):
        self.indexes = np.arange(self.num_data)
        if self.shuffle:
            np.random.shuffle(self.indexes)
        # self.edges = random_walk_sampling(self.adj_matrix, self.nodes, self.num_walks, self.walk_length)
        self.adj_list = construct_adjlist(self.adj_matrix, self.nodes, self.max_degree)

    def __len__(self):
        return int(np.floor(self.num_data / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        edges = self.edges[indexes]
        nodes_1 = edges[:, 0]
        nodes_2 = edges[:, 1]
        nodes_neg = negative_edge_sampling(nodes_2, self.sampling_array, self.num_neg_samples)

        convolution_samples_1 = convolution_sampling(self.adj_list, nodes_1, self.layer_infos)
        convolution_samples_2 = convolution_sampling(self.adj_list, nodes_2, self.layer_infos)
        convolution_samples_neg = convolution_sampling(self.adj_list, nodes_neg, self.layer_infos)

        return convolution_samples_1, convolution_samples_2, convolution_samples_neg


class UnsupervisedNodeGenerator(Sequence):
    def __init__(self, adj_matrix, edge_list, layer_infos, max_degree=10, num_neg_samples=20,
                 num_walks=50, walk_length=5, batch_size=256, shuffle=True):
        self.nodes = np.unique(edge_list)
        self.adj_matrix = adj_matrix
        self.layer_infos = layer_infos
        self.num_walks = num_walks
        self.walk_length = walk_length

        self.max_degree = max_degree
        self.batch_size = batch_size
        self.num_neg_samples = num_neg_samples
        self.shuffle = shuffle

        self.num_data = len(self.nodes)
        self.indexes = np.arange(self.num_data)

        self.sampling_array = prepare_sampling_distribution(self.adj_matrix, self.nodes, smoothing=0.75)
        self.adj_list = construct_adjlist(self.adj_matrix, self.nodes, self.max_degree)

    def on_epoch_end(self):
        self.indexes = np.arange(self.num_data)
        if self.shuffle:
            np.random.shuffle(self.indexes)
        self.adj_list = construct_adjlist(self.adj_matrix, self.nodes, self.max_degree)

    def __len__(self):
        return int(np.floor(self.num_data / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        nodes_1 = self.nodes[indexes]
        nodes_2 = self.nodes[indexes]
        nodes_neg = self.nodes[:5]

        convolution_samples_1 = convolution_sampling(self.adj_list, nodes_1, self.layer_infos)
        convolution_samples_2 = convolution_sampling(self.adj_list, nodes_2, self.layer_infos)
        convolution_samples_neg = convolution_sampling(self.adj_list, nodes_neg, self.layer_infos)

        return convolution_samples_1, convolution_samples_2, convolution_samples_neg


