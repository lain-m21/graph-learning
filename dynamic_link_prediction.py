import time
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score

from .graph_utils import sample_negative_links


def dynamic_link_prediction(model, adj_matrix_past, edge_list_past, edge_list_future, task_config):
    mode = task_config['mode']
    scaled = task_config['scaled']
    clf_type = task_config['clf_type']
    n_splits = task_config['n_splits']
    random_state = task_config['random_state']
    embeddings_savepath = task_config['embeddings_savepath']
    use_pretrained_embeddings = task_config['use_pretrained_embeddings']

    np.random.seed(random_state)

    since = time.time()

    print('Compute node embeddings on a past graph.')
    if not use_pretrained_embeddings:
        model.initialize(adj_matrix_past, edge_list_past)
        model.build()
        embeddings = model.learn_embeddings(embeddings_savepath)
    else:
        embeddings = model.load_embeddings(embeddings_savepath)

    print('--- complete ---')
    print(f'Elapsed: {time.time() - since:.4f}\n')

    print('Sample negative links based on a future graph.')
    edge_list_total = np.concatenate([edge_list_past, edge_list_future], axis=0)
    neg_ratio = len(edge_list_future) / len(edge_list_total)
    negative_links = sample_negative_links(edge_list_total, neg_ratio)
    print('--- complete ---')
    print(f'Elapsed: {time.time() - since:.4f}\n')

    edges = np.concatenate([edge_list_future, negative_links], axis=0)
    labels = np.concatenate([np.ones(len(edge_list_future), dtype=int),
                             np.zeros(len(negative_links), dtype=int)])

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores_f1 = []
    scores_ap = []
    scores_auc = []
    print(f'{n_splits:d}-fold cross validation\n')
    for index_train, index_valid in kf.split(labels):
        edges_train, edges_valid = edges[index_train], edges[index_valid]
        labels_train, labels_valid = labels[index_train], labels[index_valid]

        if mode == 'Hadamard':
            X_train = embeddings[edges_train[:, 0], :] * embeddings[edges_train[:, 1], :]
            X_valid = embeddings[edges_valid[:, 0], :] * embeddings[edges_valid[:, 1], :]
        else:
            X_train = embeddings[edges_train[:, 0], :] + embeddings[edges_train[:, 1], :]
            X_valid = embeddings[edges_valid[:, 0], :] + embeddings[edges_valid[:, 1], :]

        y_train, y_valid = labels_train, labels_valid

        if scaled:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_valid = scaler.transform(X_valid)

        if clf_type == 'logistic':
            clf = LogisticRegression(solver='saga')
        else:
            clf = LogisticRegression(solver='saga')

        print('Train a classifier.')
        clf.fit(X_train, y_train)
        print('--- complete ---')
        print(f'Elapsed: {time.time() - since:.4f}\n')

        print('Predict link labels.')
        preds_label = clf.predict(X_valid)
        preds_proba = clf.predict_proba(X_valid)[:, 1]
        print('--- complete ---')
        print(f'Elapsed: {time.time() - since:.4f}\n')

        score_f1 = f1_score(y_valid, preds_label)
        score_ap = average_precision_score(y_valid, preds_proba)
        score_auc = roc_auc_score(y_valid, preds_proba)

        print(f'F1 score: {score_f1:.6f}, Average Precision: {score_ap:.6f}, AUC: {score_auc:.6f}\n')

        scores_f1.append(score_f1)
        scores_ap.append(score_ap)
        scores_auc.append(score_auc)

    results = {
        'f1': scores_f1,
        'average_precision': scores_ap,
        'auc': scores_auc
    }

    return results
