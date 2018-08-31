import numpy as np
from numba import jit

from keras import backend as K
from keras.layers import Input, Embedding, Lambda, Concatenate, Reshape
from keras.models import Model
from keras.optimizers import Adam

from .aggregators import MeanAggregator, GCNAggregator, MeanPoolingAggregator
from .loss import build_edge_loss
from .generators import UnsupervisedEdgeGenerator


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
    """

    layer_infos: `num_samples (int)`, `output_dim (int), activation (string)`
    """

    def __init__(self, adj_matrix_train, edge_list_train, adj_matrix_valid, edge_list_valid, features, model_config):
        self.adj_matrix_train = adj_matrix_train
        self.edge_list_train = edge_list_train
        self.adj_matrix_valid = adj_matrix_valid
        self.edge_list_valid = edge_list_valid

        self.model_config = model_config
        self.layer_infos = model_config['layer_infos']

        self.support_sizes = [1]
        cum_size = 1
        for i, layer in enumerate(self.layer_infos):
            cum_size *= layer['num_samples']
            self.support_sizes.append(cum_size)
        self.support_size = cum_size

        num_nodes = adj_matrix_train.shape[0]
        feature_embed_dim = model_config['feature_dim']

        if features:
            self.features_lookup = Embedding(weights=[features], trainable=False)
        else:
            self.features_lookup = Embedding(input_dim=num_nodes + 1, output_dim=feature_embed_dim,
                                             input_length=self.support_size)

        self.batch_size = model_config['batch_size']
        self.epochs = model_config['epochs']

        loss_type = model_config['loss_type']
        neg_sample_weights = model_config['neg_sample_weights']
        self.loss_func = build_edge_loss(loss_type, neg_sample_weights)

        aggregator_type = model_config['aggregator_type']
        if aggregator_type == 'mean':
            aggregator = MeanAggregator
        elif aggregator_type == 'meanpool':
            aggregator = MeanPoolingAggregator
        else:
            aggregator = MeanAggregator

        self.aggregators = []
        self_size = 1
        for layer in self.layer_infos:
            neigh_size = self_size * layer['num_samples']
            self.aggregators.append(aggregator(layer, self_size=self_size, neigh_size=neigh_size))
            self_size = neigh_size

        self.model = None
        self.generator_train = None
        self.generator_valid = None
        self.optimizer = Adam(model_config['learning_rate'])

    def initialize_generators(self, shuffle_train=True):
        max_degree = self.model_config['max_degree']
        num_neg_samples = self.model_config['num_neg_samples']
        num_walks = self.model_config['num_walks']
        walk_length = self.model_config['walk_length']
        batch_size = self.model_config['batch_size']

        self.generator_train = UnsupervisedEdgeGenerator(adj_matrix=self.adj_matrix_train,
                                                         edge_list=self.edge_list_train,
                                                         layer_infos=self.layer_infos,
                                                         max_degree=max_degree,
                                                         num_neg_samples=num_neg_samples,
                                                         num_walks=num_walks,
                                                         walk_length=walk_length,
                                                         batch_size=batch_size,
                                                         shuffle=shuffle_train)

        self.generator_valid = UnsupervisedEdgeGenerator(adj_matrix=self.adj_matrix_valid,
                                                         edge_list=self.edge_list_valid,
                                                         layer_infos=self.layer_infos,
                                                         max_degree=max_degree,
                                                         num_neg_samples=num_neg_samples,
                                                         num_walks=num_walks,
                                                         walk_length=walk_length,
                                                         batch_size=batch_size,
                                                         shuffle=False)

    def aggregate(self, samples):
        hidden = self.features_lookup(samples)  # (batch_size, support_size, feature_embed_dim)
        dim = K.int_shape(hidden)[-1]
        for k, layer in enumerate(self.layer_infos):
            aggregator = self.aggregators[k]

            next_hidden = []
            for hop in range(len(self.layer_infos) - k):
                size_1 = self.support_sizes[hop]
                size_2 = self.support_sizes[hop + 1]
                # h_self = Lambda(lambda x: x[:, :size_1, :],
                #                 output_shape=(size_1, dim,))(hidden)
                # h_neigh = Lambda(lambda x: x[:, size_1:size_2, :],
                #                  output_shape=(size_2 - size_1, dim,))(hidden)
                h = aggregator(hidden)  # (batch_size, output_dim)
                h = Reshape((-1, layer['output_dim']))(h)
                next_hidden.append(h)  # [(batch_size, output_dim) * support_size]

            if len(next_hidden) > 1:
                hidden = Concatenate(axis=1)(next_hidden)  # (batch_size, )
            else:
                hidden = next_hidden[0]
            dim = layer['output_dim']

        outputs = Lambda(lambda x: x[:, 0, :], output_shape=(1, dim))(hidden)
        outputs = Reshape((dim,))(outputs)
        return outputs

    def build(self):
        samples_1 = Input(shape=(self.support_size,))
        samples_2 = Input(shape=(self.support_size,))
        samples_neg = Input(shape=(self.support_size,))

        outputs_1 = self.aggregate(samples_1)
        outputs_2 = self.aggregate(samples_2)
        outputs_neg = self.aggregate(samples_neg)

        outputs_1 = Reshape((-1, 1))(outputs_1)
        outputs_2 = Reshape((-1, 1))(outputs_2)
        outputs_neg = Reshape((-1, 1))(outputs_neg)

        inputs = [samples_1, samples_2, samples_neg]
        outputs = Concatenate(axis=2)([outputs_1, outputs_2, outputs_neg])

        self.model = Model(inputs, outputs)
        self.model.compile(optimizer=self.optimizer, loss=self.loss_func)

    def train(self):
        self.initialize_generators()
        self.model.fit_generator(self.generator_train, epochs=self.epochs, validation_data=self.generator_valid)

    def predict_embeddings_generator(self, generator):
        outputs = self.model.predict_generator(generator)
        embeddings = outputs[0]
        return embeddings
