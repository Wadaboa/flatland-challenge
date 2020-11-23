import torch
import numpy as np
from torch_geometric.data import Data

from flatland.core.env_observation_builder import ObservationBuilder


class GraphObservator(ObservationBuilder):

    def __init__(self, max_depth, predictor):
        super().__init__()
        self.max_depth = max_depth
        self.predictor = predictor
        self.observations = dict()
        self.observation_dim = 4

    def reset(self):
        if self.predictor is not None:
            self.predictor.reset()

    def set_env(self, env):
        super().set_env(env)
        if self.predictor:
            self.predictor.set_env(self.env)

    def get_many(self, handles=None):
        self.predictions = self.predictor.get_many()
        return super().get_many(handles)

    def get(self, handle=0):
        self.observations[handle] = self.get_graph_data(handle)
        return self.observations[handle]

    def get_graph_data(self, handle):
        # Compute edges and edges attributes
        edges = self.env.railway_encoding.get_graph_edges(
            unpacked=False, data=True
        )
        edge_index, edge_attr = [], []
        for u, v, d in edges:
            edge_index.append([
                self.env.railway_encoding.node_to_index[u],
                self.env.railway_encoding.node_to_index[v]
            ])
            edge_attr.append([
                d['weight']
            ])
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.long)

        # Compute node features
        nodes = self.env.railway_encoding.get_graph_nodes(
            unpacked=False, data=True
        )
        x = []
        for _, d in nodes:
            x.append([
                d["is_dead_end"], d["is_target"], d["is_fork"], d["is_join"]
            ])
        x = torch.tensor(x, dtype=torch.bool)

        # Create a PyTorch Geometric Data object
        data = Data(edge_index=edge_index, edge_attr=edge_attr, x=x)
        return data
