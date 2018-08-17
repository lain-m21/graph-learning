import os
import pickle
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, average_precision_score
from sklearn.linear_model import LogisticRegression


def node_classification_kfold(model, task_config):
    n_splits = task_config['n_splits']
    random_state = task_config['random_state']
    scaled = task_config['scaled']
    clf_type = task_config['clf_type']
    data_dir = task_config['data_dir']
    multilabel = task_config['multilabel']

    adj_matrix = sparse.load_npz(os.path.join(data_dir, 'adj_matrix.npz'))
    node_labels = pickle.load(open(os.path.join(data_dir, 'node_labels.pkl', 'rb')))

    print('Compute node embeddings on the train graph split.')
    model.initialize(adj_matrix)
    model.build()
    embeddings = model.learn_embeddings()
    if scaled:
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)

    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    scores_f1 = []
    scores_ap = []
    for i, (index_train, index_test) in enumerate(kf.split(node_labels)):
        X_train, X_test = embeddings[index_train], embeddings[index_test]
        y_train, y_test = node_labels[index_train], node_labels[index_test]

        if clf_type == 'logistic':
            clf = LogisticRegression(solver='saga')
        else:
            clf = LogisticRegression(solver='saga')

        if multilabel:
            clf = OneVsRestClassifier(clf, n_jobs=21)
        clf.fit(X_train, y_train)

        preds_label = clf.predict(X_test)
        preds_proba = clf.predict_proba(X_test)

        if multilabel:
            score_f1 = f1_score(y_test, preds_label, average='weighted')
            score_ap = average_precision_score(y_test, preds_proba, average='weighted')
        else:
            score_f1 = f1_score(y_test, preds_label, average='macro')
            score_ap = average_precision_score(y_test, preds_proba, average='macro')
        scores_f1.append(score_f1)
        scores_ap.append(score_ap)

    results = {
        'f1': scores_f1,
        'average_precision': scores_ap
    }

    return results
