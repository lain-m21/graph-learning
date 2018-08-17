import numpy as np
import scipy
from scipy import sparse

from .basemodel import BaseModel


class SpectralClustering(BaseModel):
    def __init__(self, config):
        super(SpectralClustering, self).__init__(config)
        self.dim = config['dim']

        self.adj_matrix = None

    def build(self):
        pass

    def learn_embeddings(self):
        n, m = self.adj_matrix.shape
        diags = self.adj_matrix.sum(axis=1).flatten()
        D = sparse.spdiags(diags, [0], m, n, format='csr')
        L = D - self.adj_matrix
        with scipy.errstate(divide='ignore'):
            diags_sqrt = 1.0 / scipy.sqrt(diags)
        diags_sqrt[scipy.isinf(diags_sqrt)] = 0
        DH = sparse.spdiags(diags_sqrt, [0], m, n, format='csr')
        laplacian = DH.dot(L.dot(DH))

        _, v = sparse.linalg.eigs(laplacian, k=self.dim + 1, which='SM')
        embeddings = v[:, 1:].real
        return embeddings

