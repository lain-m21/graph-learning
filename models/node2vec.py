import sys; sys.path.append('..')
import os
from subprocess import call
import numpy as np

from graph_utils import matrix_to_edgelist


class Node2Vec:
    def __init__(self, dim, walk_length=80, num_walks=10, context_size=10, epochs=1,
                 return_param=1, inout_param=1, directed=False, weighted=False, verbose=True, bin_path='node2vec',
                 graph_path='./data/tmp/tmp.graph', embed_path='./data/embeddings/node2vec.emb'):

        self.dim = dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.context_size = context_size
        self.epochs = epochs
        self.return_param = return_param
        self.inout_param = inout_param
        self.directed = directed
        self.weighted = weighted
        self.verbose = verbose

        self.bin_path = bin_path
        self.graph_path = graph_path
        self.embed_path = embed_path

    def initialize(self, adj_matrix, edge_list=None, edge_weights=None):
        assert edge_list
        self.save_edgelist(edge_list, self.graph_path)

    def learn_embeddings(self):
        args = [os.path.expanduser(self.bin_path)]

        args.append(f"-i:{self.graph_path}")
        args.append(f"-o:{self.embed_path}")
        args.append(f"-d:{self.dim:d}")
        args.append(f"-l:{self.walk_length:d}")
        args.append(f"-r:{self.num_walks:d}")
        args.append(f"-k:{self.context_size:d}")
        args.append(f"-e:{self.epochs:d}")
        args.append(f"-p:{self.return_param:.6f}")
        args.append(f"-q:{self.inout_param:.6f}")
        if self.directed:
            args.append("-dr")
        if self.weighted:
            args.append("-w")
        if self.verbose:
            args.append("-v")

        try:
            call(args)
        except Exception as e:
            print(str(e))
            raise Exception('node2vec not found. Please compile snap, place node2vec in the path')
        embeddings = self.load_embeddings()
        return embeddings

    def load_embeddings(self):
        with open(self.embed_path, 'r') as f:
            n, d = f.readline().strip().split()
            X = np.zeros([int(n), int(d)], dtype=np.float32)
            for line in f:
                emb = line.strip().split()
                emb_fl = [float(emb_i) for emb_i in emb[1:]]
                X[int(emb[0]), :] = emb_fl
        return X

    @staticmethod
    def save_edgelist(edge_list, path):
        lines = ''
        for edge in edge_list:
            lines += str(edge[0]) + ' ' + str(edge[1]) + '\n'
        with open(path, 'w') as f:
            f.write(lines)
