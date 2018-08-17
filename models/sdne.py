import numpy as np
from functools import reduce

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow import keras

from graph_utils import matrix_to_edgelist


class SDNE:
    def __init__(self, dim, hidden_units=(512, 256), activation='relu', beta=2, alpha=2, l2_param=0.01, batch_size=512):
        self.dim = dim
        self.hidden_units = hidden_units
        self.activation = activation
        self.beta = beta
        self.alpha = alpha
        self.l2_param = l2_param
        self.batch_size = batch_size

        self.adj_matrix = None
        self.edge_list = None
        self.edge_weights = None
        self.model = None
        self.encoder = None
        self.autoencoder = None

    def initialize(self, adj_matrix, edge_list=None, edge_weights=None):
        assert edge_list and edge_weights
        N = adj_matrix.shape[0]
        self.adj_matrix = adj_matrix.toarray().astype(np.float32)
        self.edge_list = np.concatenate([edge_list, np.flip(edge_list, 1)], axis=0)
        self.edge_weights = np.concatenate([edge_weights.reshape(-1, 1), edge_weights.reshape(-1, 1)], axis=0)

        input_a = keras.layers.Input(shape=(1,), name='input-a')
        input_b = keras.layers.Input(shape=(1,), name='input-b')
        edge_weight = keras.layers.Input(shape=(1,), name='edge_weight')

        encoding_layers = []
        decoding_layers = []

        embedding_layer = keras.layers.Embedding(input_dim=N, output_dim=N, trainable=False, input_length=1,
                                                 weights=[self.adj_matrix], name='nbr-table')
        encoding_layers.append(embedding_layer)
        encoding_layers.append(keras.layers.Flatten())

        for i, d in enumerate(self.hidden_units):
            layer = keras.layers.Dense(d, activation=self.activation, name=f'encoding-{i:d}',
                                       kernel_regularizer=keras.regularizers.l2(self.l2_param))
            encoding_layers.append(layer)

        layer = keras.layers.Dense(self.dim, activation='sigmoid', name='embeddings',
                                   kernel_regularizer=keras.regularizers.l2(self.l2_param))
        encoding_layers.append(layer)

        for i, d in enumerate(self.hidden_units[::-1]):
            layer = keras.layers.Dense(d, activation=self.activation, name=f'encoding-{i:d}',
                                       kernel_regularizer=keras.regularizers.l2(self.l2_param))
            decoding_layers.append(layer)

        layer = keras.layers.Dense(N, activation='sigmoid', name='embeddings',
                                   kernel_regularizer=keras.regularizers.l2(self.l2_param))
        decoding_layers.append(layer)

        all_layers = encoding_layers + decoding_layers

        encoded_a = reduce(lambda arg, f: f(arg), encoding_layers, input_a)
        encoded_b = reduce(lambda arg, f: f(arg), encoding_layers, input_b)
        decoded_a = reduce(lambda arg, f: f(arg), all_layers, input_a)
        decoded_b = reduce(lambda arg, f: f(arg), all_layers, input_b)

        embedding_diff = tf.subtract(encoded_a, encoded_b)
        embedding_diff = tf.multiply(embedding_diff, edge_weight)

        self.model = Model([input_a, input_b, edge_weight], [decoded_a, decoded_b, embedding_diff])

        reconstruction_loss = self.build_reconstruction_loss(self.beta)
        self.model.compile(optimizer=tf.train.AdadeltaOptimizer(),
                           loss=[reconstruction_loss, reconstruction_loss, self.edge_wise_loss],
                           loss_weights=[1, 1, self.alpha])

        self.encoder = Model(input_a, encoded_a)
        self.autoencoder = Model(input_a, decoded_a)
        self.autoencoder.compile(optimizer=tf.train.AdadeltaOptimizer, loss=reconstruction_loss)

    def learn_embeddings(self):
        self.pretrain()
        gen = self.generator()
        self.model.fit_generator(gen)

        nodes = np.arange(self.adj_matrix.shape[0])
        embeddings = self.encoder.predict(nodes)
        return embeddings

    def pretrain(self):
        assert self.autoencoder

        nodes = np.arange(self.adj_matrix.shape[0])
        node_neighbors = self.adj_matrix[nodes]
        self.autoencoder.fit(nodes.reshape(-1, 1), node_neighbors, batch_size=self.batch_size, epochs=1)

    def generator(self):
        m = len(self.edge_list)
        while True:
            for i in range(np.ceil(m / self.batch_size)):
                indices = slice(i * self.batch_size, (i + 1) * self.batch_size)
                nodes_a = self.edge_list[indices, 0].reshape(-1, 1)
                nodes_b = self.edge_list[indices, 1].reshape(-1, 1)
                neighbors_a = self.adj_matrix[nodes_a.flatten()]
                neighbors_b = self.adj_matrix[nodes_b.flatten()]
                weights = self.edge_weights[indices]

                dummy_embeddings = np.zeros([nodes_a.shape[0], self.dim])

                yield [nodes_a, nodes_b, weights], [neighbors_a, neighbors_b, dummy_embeddings]

    @staticmethod
    def build_reconstruction_loss(beta):
        assert beta > 1

        def reconstruction_loss(y_true, y_pred):
            diff = tf.square(y_true - y_pred)
            weight = y_true * (beta - 1) + 1
            weighted_diff = weight * diff
            return tf.reduce_mean(tf.reduce_sum(weighted_diff, axis=1))

        return reconstruction_loss

    @staticmethod
    def edge_wise_loss(y_true, y_pred):
        return tf.reduce_mean(tf.reduce_sum(tf.square(y_pred), axis=1))
