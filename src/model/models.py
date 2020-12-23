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
                (states.shape[0],), dtype=torch.bool,
                device=self.device
            )
        mask = torch.flatten(mask)
        mask_q = torch.zeros(
            (states.shape[0], self.action_size), device=self.device)
        q_values = self.fc(states[mask, :])
        mask_q[mask, :] = q_values
        return mask_q


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
                (states.shape[0],), dtype=torch.bool,
                device=self.device
            )
        mask = torch.flatten(mask)
        mask_val = torch.zeros(
            (states.shape[0], self.action_size),
            device=self.device
        )
        val = self.fc(states[mask, :])
        mask_val[mask, :] = val
        mask_adv = super().forward(states, mask=mask)
        mask_agg = torch.zeros((states.shape[0], 1), device=self.device)
        agg = (
            mask_adv[mask, :].mean(dim=1, keepdim=True) if self.aggregation == "mean"
            else mask_adv[mask, :].max(dim=1, keepdim=True)
        )
        mask_agg[mask, :] = agg
        return mask_val + mask_adv - mask_agg


######################################################################
########################## Entire graph GNN ##########################
######################################################################

class EntireGNN(nn.Module):
    '''
    Entire graph GNN
    '''

    def __init__(self, state_size, pos_size, embedding_size, nonlinearity="tanh",
                 gnn_hidden_size=16, depth=3, dropout=0.0, device="cpu"):
        super(EntireGNN, self).__init__()
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

    def forward(self, state, **kwargs):
        graphs = state.to_data_list()
        embs = torch.empty(
            size=(
                len(graphs) * self.pos_size,
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
                    embs[i + j] = torch.tensor(
                        [-self.depth] * self.embedding_size,
                        dtype=torch.float, device=self.device
                    )
                else:
                    embs[i + j] = emb[p.item()]

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
        super(MultiGNN, self).__init__()
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

    def forward(self, states, **kwargs):
        # Encode the FOV observation of each agent
        # with the convolutional encoder
        encoded = self.convs(states.states)

        # Use an MLP from the encoded values to have a
        # consistent number of features
        flattened = torch.flatten(encoded, start_dim=1)
        features = self.mlp(flattened)

        # Compute embeddings for each node by performing graph convolutions
        return self.gnn_conv(
            features, states.edge_index, states.edge_weight
        )
