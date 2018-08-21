import os
import numpy as np
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

from graph_utils import matrix_to_edgelist, split_graph, edgelist_to_matrix, sample_negative_links


def link_prediction(model, task_config):
    mode = task_config['mode']
    test_ratio = task_config['mode']
    neg_ratio = task_config['neg_ratio']
    n_trials = task_config['n_trials']
    random_state = task_config['random_state']
    scaled = task_config['scaled']
    clf_type = task_config['clf_type']
    data_dir = task_config['data_dir']

    adj_matrix = sparse.load_npz(os.path.join(data_dir, 'adj_matrix.npz'))

    edge_list, _ = matrix_to_edgelist(adj_matrix)

    np.random.seed(random_state)
    scores_f1 = []
    scores_ap = []
    scores_auc = []
    for i in range(n_trials):
        print(f'Trial {i+1:d} / {n_trials:d}')
        print('Split the graph into train and test.')
        edge_list_train, edge_list_test = split_graph(edge_list, test_ratio)
        adj_matrix_train = edgelist_to_matrix(edge_list_train)

        print('Compute node embeddings on the train graph split.')
        model.initialize(adj_matrix_train)
        model.build()
        embeddings = model.learn_embeddings()

        print('Sample negative links for the train graph split.')
        negative_links = sample_negative_links(edge_list_train, neg_ratio)
        edges_train = np.concatenate([edge_list_train, negative_links], axis=0)
        labels_train = np.concatenate([np.ones(len(edge_list_train), dtype=int),
                                       np.zeros(len(negative_links), dtype=int)])
        index_shuffle = np.random.permutation(np.arange(len(edges_train)))
        edges_train, labels_train = edges_train[index_shuffle], labels_train[index_shuffle]

        print('Sample negative links for the test graph split.')
        negative_links = sample_negative_links(edge_list_test, neg_ratio)
        edges_test = np.concatenate([edge_list_test, negative_links], axis=0)
        labels_test = np.concatenate([np.ones(len(edge_list_test), dtype=int),
                                      np.zeros(len(negative_links), dtype=int)])

        if mode == 'Hadamard':
            X_train = embeddings[edges_train[:, 0], :] * embeddings[edges_train[:, 1], :]
            X_test = embeddings[edges_test[:, 0], :] * embeddings[edges_test[:, 1], :]
        else:
            X_train = embeddings[edges_train[:, 0], :] + embeddings[edges_train[:, 1], :]
            X_test = embeddings[edges_test[:, 0], :] + embeddings[edges_test[:, 1], :]

        if scaled:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        y_train, y_test = labels_train, labels_test

        print('Train a classifier.')
        if clf_type == 'logistic':
            clf = LogisticRegression(solver='saga')
        else:
            clf = LogisticRegression(solver='saga')
        clf.fit(X_train, y_train)

        print('Make link predictions.')
        preds_label = clf.predict(X_test)
        preds_proba = clf.predict_proba(X_test)[:, 1]

        score_f1 = f1_score(y_test, preds_label)
        score_ap = average_precision_score(y_test, preds_proba)
        score_auc = roc_auc_score(y_test, preds_proba)

        scores_f1.append(score_f1)
        scores_ap.append(score_ap)
        scores_auc.append(score_auc)

        print()

    results = {
        'f1': scores_f1,
        'average_precision': scores_ap,
        'auc': scores_auc
    }

    return results
