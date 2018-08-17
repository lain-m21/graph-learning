import numpy as np
import tensorflow as tf

from graph_utils import sample_negative_links, matrix_to_edgelist


class LINE:
    def __init__(self, dim, batch_size=512, epochs=100, neg_ratio=5):
        self.dim = dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.neg_ratio = neg_ratio

        self.adj_matrix = None
        self.edge_list = None
        self.edge_weights = None
        self.model = None
        self.encoder = None

    def initialize(self, adj_matrix, edge_list=None, edge_weights=None):
        assert edge_list and edge_weights
        N = adj_matrix.shape[0]
        self.adj_matrix, self.edge_list, self.edge_weights = adj_matrix, edge_list, edge_weights

        input_left = tf.keras.layers.Input(shape=(1,))
        input_right = tf.keras.layers.Input(shape=(1,))

        embedding_left = tf.keras.layers.Embedding(input_dim=N + 1, output_dim=self.dim,
                                                   input_length=1, mask_zero=False)(input_left)
        embedding_right = tf.keras.layers.Embedding(input_dim=N + 1, output_dim=self.dim,
                                                    input_length=1, mask_zero=False)(input_right)
        embedding_left = tf.keras.layers.Flatten()(embedding_left)
        embedding_right = tf.keras.layers.Flatten()(embedding_right)

        product = tf.keras.layers.dot([embedding_left, embedding_right], 1)

        self.model = tf.keras.models.Model([input_left, input_right], product)
        self.encoder = tf.keras.models.Model([input_left, input_right], [embedding_left, embedding_right])

        self.model.compile(optimizer=tf.train.RMSPropOptimizer(0.01), loss=[self.line_loss])

    def learn_embeddings(self):
        gen = self.generator()
        self.model.fit_generator(gen, epochs=self.epochs)

    def generator(self):
        edge_list_all = matrix_to_edgelist(self.adj_matrix)
        negative_links = sample_negative_links(edge_list_all, self.neg_ratio)
        edges = np.concatenate([self.edge_list, negative_links], axis=0)
        weights = np.concatenate([self.edge_weights, np.zeros(len(negative_links), dtype=np.float32)], axis=0)
        index_shuffle = np.random.permutation(np.arange(len(edges)))
        edges = edges[index_shuffle]
        weights = weights[index_shuffle]
        m = len(edges)

        while True:
            for i in range(np.ceil(m / self.batch_size)):
                indices = slice(i * self.batch_size, (i + 1) * self.batch_size)
                nodes_left = edges[indices, 0].reshape(-1, 1)
                nodes_right = edges[indices, 1].reshape(-1, 1)
                relations_y = weights[indices].reshape(-1, 1)

                yield [nodes_left, nodes_right], [relations_y]

    @staticmethod
    def line_loss(y_true, y_pred):
        coeff = y_true * 2 - 1
        return tf.reduce_mean(tf.log(tf.sigmoid(coeff * y_pred)))
