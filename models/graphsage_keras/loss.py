from keras import backend as K
from keras.layers import Lambda, Flatten


def affinity(y_true, y_pred):
    node_1 = Lambda(lambda x: x[:, 0, :], output_shape=(-1,))(y_true)
    node_2 = Lambda(lambda x: x[:, 1, :], output_shape=(-1,))(y_true)
    # node_1, node_2, _ = y_pred  # (batch_size, output_dim), (batch_size, output_dim), _
    return K.sum(node_1 * node_2, axis=1)  # (batch_size, )


def negative_affinity(y_true, y_pred):
    node_1 = Lambda(lambda x: x[:, 0, :], output_shape=(-1,))(y_true)
    node_neg = Lambda(lambda x: x[:, 2, :], output_shape=(-1,))(y_true)
    # node_1, _, node_neg = y_pred  # (batch_size, output_dim), _, (neg_sample_size, output_dim)
    neg_aff = K.dot(node_1, K.transpose(node_neg))  # (batch_size, neg_sample_size)
    return neg_aff


def build_edge_loss(loss_type, neg_sample_weights=1.0):
    if loss_type == 'cross_entropy':
        def edge_loss(y_true, y_pred):
            aff = affinity(y_true, y_pred)
            neg_aff = negative_affinity(y_true, y_pred)
            true_cross_entropy = K.binary_crossentropy(aff, K.ones_like(aff))
            negative_cross_entropy = K.binary_crossentropy(neg_aff, K.zeros_like(neg_aff))
            loss = K.sum(true_cross_entropy) + neg_sample_weights * K.sum(negative_cross_entropy)
            return loss

    elif loss_type == 'skipgram':
        def edge_loss(y_true, y_pred):
            aff = affinity(y_true, y_pred)
            neg_aff = negative_affinity(y_true, y_pred)
            neg_cost = K.log(K.sum(K.exp(neg_aff), axis=1))
            loss = K.sum(aff - neg_cost)
            return loss

    else:
        def edge_loss(y_true, y_pred):
            aff = affinity(y_true, y_pred)
            neg_aff = negative_affinity(y_true, y_pred)
            true_cross_entropy = K.binary_crossentropy(aff, K.ones_like(aff))
            negative_cross_entropy = K.binary_crossentropy(neg_aff, K.zeros_like(neg_aff))
            loss = K.sum(true_cross_entropy) + neg_sample_weights * K.sum(negative_cross_entropy)
            return loss

    return edge_loss