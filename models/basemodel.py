from graph_utils import matrix_to_edgelist


class BaseModel:
    def __init__(self, config):
        self.name = config['name']

        self.adj_matrix = None
        self.edge_list = None
        self.edge_weights = None

    def initialize(self, adj_matrix):
        self.adj_matrix = adj_matrix
        self.edge_list, self.edge_weights = matrix_to_edgelist(adj_matrix)

    def build(self):
        raise NotImplementedError

    def learn_embeddings(self):
        raise NotImplementedError

