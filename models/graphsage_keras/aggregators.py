from keras import backend as K
from keras.layers import Dense, Dropout
from keras.engine.topology import Layer
from keras.initializers import glorot_uniform, zeros
from keras import activations


class MeanAggregator(Layer):
    def __init__(self, output_dim, activation, concat, use_bias, **kwargs):
        self.output_dim = output_dim
        self.activation = activations.get(activation)
        self.concat = concat
        self.use_bias = use_bias
        super(MeanAggregator, self).__init__(**kwargs)

        self.vars = {}

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.vars = {
            'neigh_weights': self.add_weight(name='neigh_weights',
                                             shape=(input_dim, self.output_dim),
                                             initializer=glorot_uniform),
            'self_weights': self.add_weight(name='self_weights',
                                            shape=(input_dim, self.output_dim),
                                            initializer=glorot_uniform),
            'bias': self.add_weight(name='bias', shape=(self.output_dim,), initializer=zeros)
        }

        super(MeanAggregator, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        self_vecs, neigh_vecs = inputs

        neigh_means = K.mean(neigh_vecs, axis=1)

        from_neighs = K.dot(neigh_means, self.vars['neight_weights'])
        from_self = K.dot(self_vecs, self.vars['self_weights'])

        if self.concat:
            output = K.concatenate([from_self, from_neighs], axis=1)
        else:
            output = K.bias_add(from_self, from_neighs, data_format='channels_last')

        if self.use_bias:
            output = K.bias_add(output, self.vars['bias'], data_format='channels_last')

        return self.activation(output)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'activation': activations.serialize(self.activation),
            'concat': self.concat,
            'use_bias': self.use_bias
        }
        base_config = super(MeanAggregator, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GCNAggregator(Layer):
    def __init__(self, output_dim, activation, use_bias, **kwargs):
        self.output_dim = output_dim
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        super(GCNAggregator, self).__init__(**kwargs)

        self.vars = {}

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.vars = {
            'weights': self.add_weight(name='neigh_weights',
                                       shape=(input_dim, self.output_dim),
                                       initializer=glorot_uniform),
            'bias': self.add_weight(name='bias', shape=(self.output_dim,), initializer=zeros)
        }

        super(GCNAggregator, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        self_vecs, neigh_vecs = inputs

        self_vecs = K.expand_dims(self_vecs, axis=1)
        means = K.mean(K.concatenate([neigh_vecs, self_vecs], axis=1), axis=1)

        output = K.dot(means, self.vars['weights'])

        if self.use_bias:
            output = K.bias_add(output, self.vars['bias'], data_format='channels_last')

        return self.activation(output)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias
        }
        base_config = super(GCNAggregator, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MeanPoolingAggregator(Layer):
    def __init__(self, output_dim, hidden_dim, num_dense_layers, activation, concat, use_bias, dropout, **kwargs):
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_dense_layers = num_dense_layers
        self.activation = activations.get(activation)
        self.concat = concat
        self.use_bias = use_bias
        self.dropout = dropout
        super(MeanPoolingAggregator, self).__init__(**kwargs)

        self.vars = {}
        self.dense_layers = []
        self.input_dim = None

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.input_dim = input_dim

        self.vars = {
            'neigh_weights': self.add_weight(name='neigh_weights',
                                             shape=(self.hidden_dim, self.output_dim),
                                             initializer=glorot_uniform),
            'self_weights': self.add_weight(name='self_weights',
                                            shape=(input_dim, self.output_dim),
                                            initializer=glorot_uniform),
            'bias': self.add_weight(name='bias', shape=(self.output_dim,), initializer=zeros)
        }

        for _ in range(self.num_dense_layers):
            self.dense_layers.append(Dense(self.hidden_dim, activation=self.activation))
            self.dense_layers.append(Dropout(self.dropout))
        if self.num_dense_layers:
            self.dense_layers.append(Dense(self.hidden_dim, activation=self.activation))

        super(MeanPoolingAggregator, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        self_vecs, neigh_vecs = inputs

        sess = K.get_session()
        dims = K.shape(neigh_vecs).eval(session=sess)
        batch_size = dims[0]
        num_neighbors = dims[1]
        h_reshaped = K.reshape(neigh_vecs, (batch_size * num_neighbors, self.input_dim))

        for layer in self.dense_layers:
            h_reshaped = layer(h_reshaped)

        neigh_h = K.reshape(h_reshaped, (batch_size, num_neighbors, self.hidden_dim))
        neigh_h = K.mean(neigh_h, axis=1)

        from_neighs = K.dot(neigh_h, self.vars['neight_weights'])
        from_self = K.dot(self_vecs, self.vars['self_weights'])

        if self.concat:
            output = K.concatenate([from_self, from_neighs], axis=1)
        else:
            output = K.bias_add(from_self, from_neighs, data_format='channels_last')

        if self.use_bias:
            output = K.bias_add(output, self.vars['bias'], data_format='channels_last')

        return self.activation(output)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'hidden_dim': self.hidden_dim,
            'num_dense_layers': self.num_dense_layers,
            'activation': activations.serialize(self.activation),
            'concat': self.concat,
            'use_bias': self.use_bias,
            'dropout': self.dropout
        }
        base_config = super(MeanPoolingAggregator, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
