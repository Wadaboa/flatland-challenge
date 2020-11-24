import torch
import numpy as np
from torch_geometric.data import Data

from flatland.core.env_observation_builder import ObservationBuilder

import env_utils


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
        edge_index, edge_weight = [], []
        for u, v, d in edges:
            edge_index.append([
                self.env.railway_encoding.node_to_index[u],
                self.env.railway_encoding.node_to_index[v]
            ])
            edge_weight.append(d['weight'])
        edge_index = torch.tensor(
            edge_index, dtype=torch.long
        ).t().contiguous()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)

        # Compute node features
        nodes = self.env.railway_encoding.get_graph_nodes(
            unpacked=False, data=True
        )
        x = [None] * len(nodes)
        for n, d in nodes:
            x[self.env.railway_encoding.node_to_index[n]] = [
                d["is_dead_end"], d["is_target"], d["is_fork"], d["is_join"]
            ]
        x = torch.tensor(x, dtype=torch.float)

        # Store a list of important positions, so that the DQN is called with
        # the GNN embeddings of these nodes
        agent_position = self.env.railway_encoding.get_agent_cell(handle)
        agent_pos_index = -1
        successors = []
        if agent_position is not None:
            agent_in_packed = self.env.railway_encoding.is_node(
                agent_position, unpacked=False
            )
            if agent_in_packed:
                successors = self.env.railway_encoding.get_successors(
                    agent_position, unpacked=False
                )
            else:
                actual_agent_position = tuple(agent_position)
                agent_position, _ = self.env.railway_encoding.previous_node(
                    actual_agent_position
                )
                successor, _ = self.env.railway_encoding.next_node(
                    actual_agent_position
                )
                successors = [successor]
            agent_pos_index = self.env.railway_encoding.node_to_index[agent_position]

        successors_indexes = {"left": -1, "right": -1}
        for succ in successors:
            succ_index = self.env.railway_encoding.node_to_index[succ]
            succ_choice = self.env.railway_encoding.get_edge_data(
                agent_position, succ, 'choice', unpacked=False
            )
            if succ_choice == env_utils.RailEnvChoices.CHOICE_LEFT:
                successors_indexes["left"] = succ_index
            elif succ_choice == env_utils.RailEnvChoices.CHOICE_RIGHT:
                successors_indexes["right"] = succ_index
        pos = torch.tensor([
            successors_indexes["left"],
            successors_indexes["right"],
            agent_pos_index
        ], dtype=torch.long)

        # Create a PyTorch Geometric Data object
        data = Data(
            edge_index=edge_index, edge_weight=edge_weight, pos=pos, x=x
        )
        return data