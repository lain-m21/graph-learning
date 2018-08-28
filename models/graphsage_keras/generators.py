import numpy as np
from numba import jit
from keras.utils import Sequence


class UnsupervisedGenerator(Sequence):
    def __init__(self, adj_matrix, edge_list, max_degree, context_pairs=None, batch_size=256, shuffle=True):
        self.nodes = np.unique(edge_list)
        if context_pairs:
            self.edges = context_pairs
        else:
            self.edges = edge_list

        self.max_degree = max_degree
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.embedding = embedding
        self.X_input = X_input
        self.batch_size = batch_size
        self.shuffle = shuffle

        if isinstance(self.X_input, dict):
            self.num_data = self.X_input['num_data']
        else:
            self.num_data = len(self.X_input)
        self.indexes = np.arange(self.num_data)
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(self.num_data)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(self.num_data / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        if isinstance(self.X_input, dict):
            X = {}
            for key in self.X_input.keys():
                if key == 'num_data':
                    continue
                X[key] = self.X_input[key][indexes]
        else:
            X = self.X_input[indexes]
        y = self.embedding.predict(X)
        return X, y


class EdgeMinibatchSampler:
    def __init__(self, adj_matrix, edge_list, placeholders, context_pairs=None,
                 batch_size=100, max_degree=25, shuffle=True):

        self.nodes = np.unique(edge_list)
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.batch_num = 0

        if context_pairs:
            self.edges = context_pairs

        if shuffle:
            self.nodes = np.random.permutation(self.nodes)
            self.edges = np.random.permutation(self.edges)

        self.adj, self.deg = construct_adj(adj_matrix, self.edges, self.max_degree)

    def end(self):
        return self.batch_num * self.batch_size >= len(self.edges)

    def batch_feed_dict(self, batch_edges):
        batch1 = []
        batch2 = []
        for node1, node2 in batch_edges:
            batch1.append(node1)
            batch2.append(node2)

        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size']: len(batch_edges)})
        feed_dict.update({self.placeholders['batch1']: batch1})
        feed_dict.update({self.placeholders['batch2']: batch2})

        return feed_dict

    def incremental_embed_feed_dict(self, size, iter_num):
        node_list = self.nodes
        val_nodes = node_list[iter_num * size:min((iter_num + 1) * size,
                                                  len(node_list))]
        val_edges = [(n, n) for n in val_nodes]
        return self.batch_feed_dict(val_edges), (iter_num + 1) * size >= len(node_list), val_edges

    def next_minibatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.edges))
        batch_edges = self.edges[start_idx: end_idx]
        return self.batch_feed_dict(batch_edges)

    def num_training_batches(self):
        return len(self.edges) // self.batch_size + 1

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        self.edges = np.random.permutation(self.edges)
        self.nodes = np.random.permutation(self.nodes)
        self.batch_num = 0