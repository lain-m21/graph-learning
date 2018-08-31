import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression


def node_classification(model, adj_matrix, edge_list, node_labels, task_config):
    scaled = task_config['scaled']
    clf_type = task_config['clf_type']
    n_splits = task_config['n_splits']
    random_state = task_config['random_state']
    embeddings_savepath = task_config['embeddings_savepath']
    use_pretrained_embeddings = task_config['use_pretrained_embeddings']

    np.random.seed(random_state)
    since = time.time()

    print('Compute node embeddings on the train graph split.')
    if not use_pretrained_embeddings:
        model.initialize(adj_matrix, edge_list)
        model.build()
        embeddings = model.learn_embeddings(embeddings_savepath)
    else:
        embeddings = model.load_embeddings(embeddings_savepath)
    print('--- complete ---')
    print(f'Elapsed: {time.time() - since:.4f}')

    if scaled:
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)

    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    scores_f1 = []
    print('Stratified KFold shuffle split')
    for i, (index_train, index_test) in enumerate(skf.split(node_labels, node_labels)):
        X_train, X_test = embeddings[index_train], embeddings[index_test]
        y_train, y_test = node_labels[index_train], node_labels[index_test]

        if clf_type == 'logistic':
            clf = LogisticRegression(solver='saga')
        else:
            clf = LogisticRegression(solver='saga')

        print('Train a classifier')
        clf.fit(X_train, y_train)
        print('--- complete ---')
        print(f'Elapsed: {time.time() - since:.4f}')

        print('Make predictions')
        preds_label = clf.predict(X_test)

        score_f1 = f1_score(y_test, preds_label, average='macro')
        scores_f1.append(score_f1)
        print('--- complete ---')
        print(f'Elapsed: {time.time() - since:.4f}')
        print(f'F1 score: {score_f1:.6f}')

    results = {
        'f1': scores_f1
    }

    return results


def node_classification_multilabel(model, adj_matrix, edge_list, node_labels, task_config):
    scaled = task_config['scaled']
    clf_type = task_config['clf_type']
    n_splits = task_config['n_splits']
    random_state = task_config['random_state']
    embeddings_savepath = task_config['embeddings_savepath']
    use_pretrained_embeddings = task_config['use_pretrained_embeddings']

    np.random.seed(random_state)
    since = time.time()

    print('Compute node embeddings on the train graph split.')
    if not use_pretrained_embeddings:
        model.initialize(adj_matrix, edge_list)
        model.build()
        embeddings = model.learn_embeddings(embeddings_savepath)
    else:
        embeddings = model.load_embeddings(embeddings_savepath)
    print('--- complete ---')
    print(f'Elapsed: {time.time() - since:.4f}')

    if scaled:
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)

    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    scores_f1 = []
    print('Stratified KFold shuffle split')
    for i, (index_train, index_test) in enumerate(kf.split(node_labels)):
        X_train, X_test = embeddings[index_train], embeddings[index_test]
        y_train, y_test = node_labels[index_train], node_labels[index_test]

        if clf_type == 'logistic':
            clf = LogisticRegression(solver='saga')
        else:
            clf = LogisticRegression(solver='saga')

        clf = OneVsRestClassifier(clf, n_jobs=21)

        print('Train a classifier')
        clf.fit(X_train, y_train)
        print('--- complete ---')
        print(f'Elapsed: {time.time() - since:.4f}')

        print('Make predictions')
        preds_label = clf.predict(X_test)

        score_f1 = f1_score(y_test, preds_label, average='weighted')
        scores_f1.append(score_f1)
        print('--- complete ---')
        print(f'Elapsed: {time.time() - since:.4f}')
        print(f'F1 score: {score_f1:.6f}')

    results = {
        'f1': scores_f1
    }

    return results


def node_classification_graphconvolution(model, adj_matrix_train, adj_matrix_valid, edge_list_train, edge_list_valid,
                                         node_labels, task_config):
    scaled = task_config['scaled']
    clf_type = task_config['clf_type']
    n_splits = task_config['n_splits']
    random_state = task_config['random_state']
    embeddings_savepath = task_config['embeddings_savepath']
    use_pretrained_embeddings = task_config['use_pretrained_embeddings']

    np.random.seed(random_state)
    since = time.time()

    print('Compute node embeddings on the train graph split.')
    if not use_pretrained_embeddings:
        model.initialize(adj_matrix_train, edge_list_train)
        model.set_valid_set(adj_matrix_valid, edge_list_valid)
        model.build()
        embeddings = model.learn_embeddings(embeddings_savepath)
    else:
        embeddings = model.load_embeddings(embeddings_savepath)
    print('--- complete ---')
    print(f'Elapsed: {time.time() - since:.4f}')

    if scaled:
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)

    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    scores_f1 = []
    print('Stratified KFold shuffle split')
    for i, (index_train, index_test) in enumerate(skf.split(node_labels, node_labels)):
        X_train, X_test = embeddings[index_train], embeddings[index_test]
        y_train, y_test = node_labels[index_train], node_labels[index_test]

        if clf_type == 'logistic':
            clf = LogisticRegression(solver='saga')
        else:
            clf = LogisticRegression(solver='saga')

        print('Train a classifier')
        clf.fit(X_train, y_train)
        print('--- complete ---')
        print(f'Elapsed: {time.time() - since:.4f}')

        print('Make predictions')
        preds_label = clf.predict(X_test)

        score_f1 = f1_score(y_test, preds_label, average='macro')
        scores_f1.append(score_f1)
        print('--- complete ---')
        print(f'Elapsed: {time.time() - since:.4f}')
        print(f'F1 score: {score_f1:.6f}')

    results = {
        'f1': scores_f1
    }

    return results
