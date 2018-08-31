from keras import backend as K
from keras.layers import Dense, Dropout, Lambda
from keras.engine.topology import Layer
from keras.initializers import glorot_uniform, zeros
from keras import activations


class MeanAggregator(Layer):
    def __init__(self, layer, **kwargs):
        self.output_dim = layer['output_dim']
        self.activation = activations.get(layer['activation'])
        self.concat = layer['concat']
        self.use_bias = layer['use_bias']
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
            output = from_self + from_neighs

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
    def __init__(self, layer, **kwargs):
        self.output_dim = layer['output_dim']
        self.activation = activations.get(layer['activation'])
        self.use_bias = layer['use_bias']
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
    def __init__(self, layer, self_size, neigh_size, **kwargs):
        self.output_dim = layer['output_dim']
        self.hidden_dim = layer['hidden_dim']
        self.num_dense_layers = layer['num_dense_layers']
        self.activation = activations.get(layer['activation'])
        self.concat = layer['concat']
        self.use_bias = layer['use_bias']
        self.dropout = layer['dropout']
        super(MeanPoolingAggregator, self).__init__(**kwargs)

        self.self_size = self_size
        self.neigh_size = neigh_size

        self.vars = {}
        self.dense_layers = []
        self.input_dim = None

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.input_dim = input_dim

        self.vars = {
            'neigh_weights': self.add_weight(shape=(self.hidden_dim, self.output_dim),
                                             initializer='glorot_uniform',
                                             name='neigh_weights'),
            'self_weights': self.add_weight(shape=(input_dim, self.output_dim),
                                            initializer='glorot_uniform',
                                            name='self_weights'),
            'bias': self.add_weight(name='bias', shape=(self.output_dim,), initializer='zeros')
        }

        for i in range(self.num_dense_layers):
            kernel = self.add_weight(shape=(input_dim, self.hidden_dim),
                                     initializer='glorot_uniform',
                                     name='dense_kernel_%d' % i)
            bias = self.add_weight(shape=(self.hidden_dim,),
                                   initializer='zeros',
                                   name='dense_bias_%d' % i)
            self.dense_layers.append({'kernel': kernel, 'bias': bias})
            input_dim = self.hidden_dim

        super(MeanPoolingAggregator, self).build(input_shape)  # Be sure to call this somewhere!

    def apply_dense_layers(self, x):
        for layer in self.dense_layers:
            x = K.dot(x, layer['kernel'])
            x = K.bias_add(x, layer['bias'], data_format='channels_last')
            x = self.activation(x)
            x = K.dropout(x, self.dropout)
        return x

    def call(self, inputs):
        self_vecs = Lambda(lambda x: x[:, :self.self_size, :],
                           output_shape=(self.self_size, -1,))(inputs)
        neigh_vecs = Lambda(lambda x: x[:, self.self_size:self.neigh_size, :],
                            output_shape=(self.neigh_size - self.self_size, -1,))(inputs)

        # input_shape = K.int_shape(inputs)
        # batch_size = input_shape[0]
        #
        # h_reshaped = K.reshape(neigh_vecs, (batch_size * self.neigh_size, self.input_dim))

        neigh_h = self.apply_dense_layers(neigh_vecs)

        # neigh_h = K.reshape(h_reshaped, (batch_size, self.neigh_size, self.hidden_dim))
        neigh_h = K.mean(neigh_h, axis=1)

        from_neighs = K.dot(neigh_h, self.vars['neigh_weights'])
        from_self = K.dot(self_vecs, self.vars['self_weights'])

        if self.concat:
            output = K.concatenate([from_self, from_neighs], axis=1)
        else:
            output = from_self + from_neighs

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
