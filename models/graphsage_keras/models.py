import numpy as np
from numba import jit

from keras import backend as K
from keras.layers import Input, Embedding, Lambda
from keras.models import Model
from keras.optimizers import Adam

from .aggregators import MeanAggregator, GCNAggregator, MeanPoolingAggregator
from .loss import build_edge_loss


@jit('Tuple((i8[:, :], i8[:]))(i8[:, :], i8[:], i8)')
def construct_adj_list(adj_matrix, nodes, max_degree):
    adj_list = len(nodes) * np.ones((len(nodes) + 1, max_degree), dtype=int)
    degrees = np.zeros((len(nodes), ), dtype=int)

    for node_id in nodes:
        neighbors = adj_matrix[node_id, :]
        degrees[node_id] = len(neighbors)
        if len(neighbors) == 0:
            continue
        if len(neighbors) > max_degree:
            neighbors = np.random.choice(neighbors, max_degree, replace=False)
        elif len(neighbors) < max_degree:
            neighbors = np.random.choice(neighbors, max_degree, replace=True)

        adj_list[node_id, :] = neighbors

    return adj_list, degrees


class GraphSAGEUnsupervised:
    def __init__(self, aggregator_configs, layer_infos, adj_list, features, loss_info):
        aggregator_type = aggregator_configs['aggregator_type']
        if aggregator_type == 'mean':
            self.aggregator = MeanAggregator
        else:
            self.aggregator = MeanAggregator

        self.layer_infos = layer_infos

        if features:
            self.features_lookup = Embedding(weights=[features], trainable=False)
        else:
            self.features_lookup = None

        self.adj_lookup = Embedding(weights=[adj_list])  # adj_list
        self.max_degree = 0  # adj_list_size

        loss_type = loss_info['loss_type']
        neg_sample_weights = loss_info['neg_sample_weights']
        self.loss_func = build_edge_loss(loss_type, neg_sample_weights)
        self.model = None
        self.optimizer = Adam()

        self.build()

    def build(self):
        nodes_1 = Input(shape=(1,))
        nodes_2 = Input(shape=(1,))
        neg_nodes = Input(shape=(1,))

        samples_1, support_sizes_1 = self.sample_for_convolution(nodes_1)
        samples_2, support_sizes_2 = self.sample_for_convolution(nodes_2)
        neg_samples, neg_support_sizes = self.sample_for_convolution(neg_nodes)

        outputs_1, aggregators = self.aggregate(samples_1, self.layer_infos)
        outputs_2, _ = self.aggregate(samples_2, self.layer_infos, aggregators)
        neg_outputs = self.aggregate(neg_samples, self.layer_infos, aggregators)

        inputs = [nodes_1, nodes_2, neg_nodes]
        outputs = [outputs_1, outputs_2, neg_outputs]

        self.model = Model(inputs, outputs)
        self.model.compile(optimizer=self.optimizer, loss=self.loss_func)

    def sample_neighbors(self, nodes, sample_size):
        neighbors = self.adj_lookup(nodes)
        index_shuffle = np.random.permutation(self.max_degree)
        neighbors = Lambda(lambda x: x[:, index_shuffle])(neighbors)
        neighbors = Lambda(lambda x: x[:, :sample_size], output_shape=(sample_size,))(neighbors)
        return neighbors

    def sample_for_convolution(self, nodes):
        samples = [nodes]
        support_size = 1
        support_sizes = [support_size]
        for k in range(len(self.layer_infos)):
            num_samples = self.layer_infos[k]['num_samples']
            support_size *= num_samples
            neighbors = self.sample_neighbors(nodes, num_samples)
            samples.append(neighbors)
            nodes = neighbors
            support_sizes.append(support_size)

        return samples, support_sizes

    def aggregate(self, samples, layer_infos, aggregators=None):
        new_aggregator = aggregators is None
        if new_aggregator:
            aggregators = []

        hidden = [self.features_lookup(nodes) for nodes in samples]
        for k, layer in enumerate(layer_infos):
            if new_aggregator:
                aggregator = self.aggregator(layer['output_dim'], layer['activation'], layer['concat'], layer['use_bias'])
                aggregators.append(layer)
            else:
                aggregator = aggregators[k]

            next_hidden = []
            for hop in range(len(layer['num_samples']) - k):
                h = aggregator((hidden[hop], hidden[hop + 1]))
                next_hidden.append(h)
            hidden = next_hidden

        return hidden[0], aggregators
