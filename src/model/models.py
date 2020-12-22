import numpy as np
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F

from model import model_utils


######################################################################
################################# DQN ################################
######################################################################

class DQN(nn.Module):
    '''
    Vanilla deep Q-Network
    '''

    def __init__(self, state_size, action_size, hidden_sizes=[128, 128],
                 nonlinearity="tanh", device="cpu"):
        super(DQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.fc = model_utils.get_linear(
            state_size, action_size, hidden_sizes, nonlinearity=nonlinearity
        )

    def forward(self, states, mask=None):
        states = torch.flatten(states, start_dim=1)
        assert len(states.shape) == 2 and states.shape[1] == self.state_size
        if mask is None:
            mask = torch.ones(
                (states.shape[0], 1), dtype=torch.bool,
                device=self.device
            )

        empty_tensor = torch.zeros((1, self.action_size), device=self.device)
        return torch.where(mask, self.fc(states), empty_tensor).to(self.device)


######################################################################
########################### Dueling DQN ##############################
######################################################################

class DuelingDQN(DQN):
    '''
    Dueling DQN
    '''

    def __init__(self, state_size, action_size, hidden_sizes=[128, 128],
                 nonlinearity="tanh", device="cpu", aggregation="mean"):
        super(DuelingDQN, self).__init__(
            state_size, action_size, hidden_sizes=hidden_sizes,
            nonlinearity=nonlinearity, device=device
        )
        self.aggregation = aggregation
        self.fc_val = model_utils.get_linear(
            state_size, 1, hidden_sizes, nonlinearity=nonlinearity
        )

    def forward(self, states, mask=None):
        states = torch.flatten(states, start_dim=1)
        assert len(states.shape) == 2 and states.shape[1] == self.state_size
        if mask is None:
            mask = torch.ones(
                (states.shape[0], 1), dtype=torch.bool,
                device=self.device
            )

        empty_tensor = torch.zeros((1, 1), device=self.device)
        val = torch.where(
            mask, self.fc_val(states), empty_tensor
        ).to(self.device)
        adv = super().forward(states, mask=mask)
        agg = (
            adv.mean(dim=1, keepdim=True) if self.aggregation == "mean"
            else adv.max(dim=1, keepdim=True)
        )
        return val + adv - agg


######################################################################
########################## Single agent GNN ##########################
######################################################################

class SingleGNN(nn.Module):
    '''
    Single agent GNN
    '''

    def __init__(self, state_size, pos_size, embedding_size, nonlinearity="tanh",
                 gnn_hidden_size=16, depth=3, dropout=0.0, device="cpu"):
        self.state_size = state_size
        self.pos_size = pos_size
        self.embedding_size = embedding_size
        self.depth = depth
        self.dropout = dropout
        self.device = device
        self.nl = nn.ReLU() if nonlinearity == "relu" else nn.Tanh()
        self.gnn_conv = nn.ModuleList()
        self.gnn_conv.append(gnn.GCNConv(
            state_size, gnn_hidden_size
        ))
        for i in range(1, self.depth - 1):
            self.gnn_conv.append(gnn.GCNConv(
                gnn_hidden_size, gnn_hidden_size
            ))
        self.gnn_conv.append(gnn.GCNConv(
            gnn_hidden_size, self.embedding_size
        ))

    def forward(self, state):
        graphs = state.to_data_list()
        embs = torch.empty(
            size=(
                len(graphs),
                self.pos_size,
                self.embedding_size
            ), dtype=torch.float,
            device=self.device
        )

        # For each graph in the batch
        for i, graph in enumerate(graphs):
            x, edge_index, edge_weight, pos = (
                graph.x, graph.edge_index, graph.edge_weight, graph.pos
            )

            # Perform a number of graph convolutions specified by
            # the given depth
            for d in range(self.depth):
                x = self.gnn_conv[d](x, edge_index, edge_weight=edge_weight)
                emb = x
                x = self.nl(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

            # Extract useful embeddings
            for j, p in enumerate(pos):
                if p == -1:
                    embs[i, j] = torch.tensor(
                        [-self.depth] * self.embedding_size,
                        dtype=torch.float, device=self.device
                    )
                else:
                    embs[i, j] = emb[p.item()]

        return embs


######################################################################
########################## Multi agent GNN ###########################
######################################################################

class MultiGNN(nn.Module):
    '''
    Multi agent GNN
    '''

    def __init__(self, input_width, input_height, input_channels, output_channels,
                 hidden_channels=[16, 32, 16], pool=False, embedding_size=128, hidden_sizes=[128, 128],
                 nonlinearity="relu", device="cpu"):
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels
        self.embedding_size = embedding_size
        self.device = device

        # Encoder
        self.convs = model_utils.get_conv(
            input_channels, output_channels, hidden_channels,
            kernel_size=3, stride=1, padding=0,
            nonlinearity=nonlinearity, pool=pool
        )

        # MLP
        output_width, output_height = model_utils.conv_block_output_size(
            self.convs, input_width, input_height
        )
        assert output_width > 0 and output_height > 0
        self.mlp = model_utils.get_linear(
            output_width * output_height * output_channels,
            embedding_size, hidden_sizes, nonlinearity=nonlinearity
        )

        # GNN
        self.gnn_conv = gnn.GCNConv(
            embedding_size, embedding_size, add_self_loops=False
        )

    def forward(self, states, adjacencies, mask=None):
        if mask is None:
            mask = torch.ones(
                (states.shape[0], states.shape[1], 1), dtype=torch.bool,
                device=self.device
            )
        active_indexes = mask.nonzero()
        embeddings = torch.empty(
            size=(
                states.shape[0],
                states.shape[1],
                self.embedding_size
            ), dtype=torch.float,
            device=self.device
        )
        for batch_number, batch in enumerate(states):
            current_active_indexes = active_indexes[
                active_indexes[:, 0] == batch_number
            ]
            # If every agent is inactive, skip computations
            if current_active_indexes.shape[0] == 0:
                continue

            # Encode the FOV observation of each agent
            # with the convolutional encoder
            encoded = self.convs(batch)

            # Use an MLP from the encoded values to have a
            # consistent number of features
            flattened = torch.flatten(encoded, start_dim=1)
            features = self.mlp(flattened)

            # Create the graph used by the defined GNN conv,
            # specified by the given adjacency matrix
            edge_index, edge_weight = [], []
            num_agents = adjacencies.shape[1]
            for i in range(num_agents):
                for j in range(num_agents):
                    if adjacencies[batch_number, i, j] != 0 or i == j:
                        edge_index.append([i, j])
                        edge_weight.append(adjacencies[batch_number, i, j])
            edge_index = torch.tensor(
                edge_index, dtype=torch.long, device=self.device
            ).t().contiguous()
            edge_weight = torch.tensor(
                edge_weight, dtype=torch.float, device=self.device
            )

            # Compute embeddings for each node by performing graph convolutions
            embeddings[batch_number] = self.gnn_conv(
                features, edge_index, edge_weight=edge_weight
            )

        # Return embeddings of size (batch_size * num_agents, embedding_size)
        return embeddings.flatten(start_dim=0, end_dim=1)
