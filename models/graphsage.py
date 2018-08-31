import numpy as np

from .basemodel import BaseModel
from .graphsage_keras.models import GraphSAGEUnsupervised
from .graphsage_keras.generators import UnsupervisedNodeGenerator


class GraphSAGE(BaseModel):
    """
    Wrapper of GraphSAGE Keras implementation (incomplete)

    """
    def __init__(self, config):
        super(GraphSAGE, self).__init__(config)

        self.model_config = config
        self.model = None
        self.adj_matrix_valid = None
        self.edge_list_valid = None

    def set_valid_set(self, adj_matrix_valid, edge_list_valid):
        self.adj_matrix_valid = adj_matrix_valid
        self.edge_list_valid = edge_list_valid

    def build(self):
        self.model = GraphSAGEUnsupervised(self.adj_matrix, self.edge_list,
                                           self.adj_matrix_valid, self.edge_list_valid,
                                           features=None, model_config=self.model_config)
        self.model.build()
        pass

    def learn_embeddings(self, save_path=None):
        self.model.train()

        embeddings = self.predict_node_embeddings()

        if save_path:
            np.save(save_path, embeddings)

        return embeddings

    def predict_node_embeddings(self):
        layer_infos = self.model_config['layer_infos']
        batch_size = self.model_config['batch_size']
        node_generator = UnsupervisedNodeGenerator(self.adj_matrix, self.edge_list, layer_infos,
                                                   batch_size=batch_size, shuffle=False)
        embeddings = self.model.predict_embeddings_generator(node_generator)
        return embeddings

    def load_embeddings(self, save_path):
        embeddings = np.load(save_path)
        return embeddings
