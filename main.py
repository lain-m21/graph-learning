import os
import json
import argparse
import numpy as np

from link_prediction import link_prediction
from node_classification import node_classification_kfold

from models import SpectralClustering, SDNE, LINE, Node2Vec


def main(args):
    model_config = json.load(open(os.path.join('./models/model_configs', args.model_config + '.json'), 'r'))
    if model_config['model'] == 'spc':
        model = SpectralClustering(model_config)
    elif model_config['model'] == 'node2vec':
        model = Node2Vec(model_config)
    elif model_config['model'] == 'sdne':
        model = SDNE(model_config)
    elif model_config['model'] == 'LINE':
        model = LINE(model_config)
    else:
        model = SpectralClustering({'dim': 128})

    task_config = json.load(open(os.path.join('./task_configs', args.task_config + '.json'), 'r'))

    if task_config['task'] == 'link_prediction':
        results = link_prediction(model, task_config)
    elif task_config['task'] == 'node_classification':
        results = node_classification_kfold(model, task_config)
    else:
        raise ValueError()

    for key, scores in results.items():
        score_mean = np.mean(scores)
        score_std = np.std(scores)
        print(f'{key} score: mean - {score_mean:.6f}, std - {score_std:.6f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', default='model_config_0')
    parser.add_argument('--task_config', default='task_config_0')
    args = parser.parse_args()

    main(args)
