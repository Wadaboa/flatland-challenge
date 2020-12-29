import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.container import ParameterList
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

    def __init__(self, state_size, action_size, params, device="cpu"):
        super(DQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.params = params
        self.device = device
        self.fc = model_utils.get_linear(
            state_size, action_size, self.params.hidden_sizes,
            nonlinearity=self.params.nonlinearity
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

    def __init__(self, state_size, action_size, params, device="cpu"):
        super(DuelingDQN, self).__init__(
            state_size, action_size, params, device=device
        )
        self.aggregation = self.params.dueling.aggregation.get_true_key()
        self.fc_val = model_utils.get_linear(
            state_size, 1, self.params.hidden_sizes,
            nonlinearity=self.params.nonlinearity
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

    def __init__(self, state_size, depth, params, device="cpu"):
        super(EntireGNN, self).__init__()
        self.state_size = state_size
        self.depth = depth
        self.params = params
        self.device = device

        self.embedding_size = self.params.embedding_size
        self.hidden_size = self.params.hidden_size
        self.pos_size = self.params.pos_size
        self.dropout = self.params.dropout
        self.nonlinearity = self.params.nonlinearity.get_true_key()

        self.nl = (
            nn.ReLU(inplace=True) if self.nonlinearity == "relu" else nn.Tanh()
        )
        self.gnn_conv = nn.ModuleList()
        sizes = (
            [state_size] +
            [self.hidden_size] * (self.depth - 2) +
            [self.embedding_size]
        )
        for i in range(1, len(sizes)):
            self.gnn_conv.append(
                gnn.GCNConv(sizes[i - 1], sizes[i])
            )

    def forward(self, states, **kwargs):
        graphs = states.to_data_list()
        embs = torch.zeros(
            size=(
                len(graphs),
                self.pos_size * self.embedding_size
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
            tmp_embs = torch.full(
                (self.pos_size, self.embedding_size),
                [-self.depth] * self.embedding_size,
                dtype=torch.float,
                device=self.device
            )
            for j, p in enumerate(pos):
                if p != -1:
                    tmp_embs[j] = emb[p.item()]
            embs[i] = torch.flatten(tmp_embs)

        return embs


######################################################################
########################## Multi agent GNN ###########################
######################################################################

class MultiGNN(nn.Module):
    '''
    Multi agent GNN
    '''

    def __init__(self, input_width, input_height, input_channels, params, device="cpu"):
        super(MultiGNN, self).__init__()
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.params = params
        self.device = device

        self.output_channels = self.params.cnn_encoder.output_channels
        self.hidden_channels = self.params.cnn_encoder.hidden_channels
        self.mlp_output_size = self.params.mlp_compression.output_size
        self.mlp_hidden_sizes = self.params.mlp_compression.hidden_sizes
        self.gnn_hidden_sizes = self.params.gnn_communication.hidden_sizes
        self.embedding_size = self.params.gnn_communication.embedding_size
        self.dropout = self.params.gnn_communication.dropout
        self.nonlinearity = self.params.nonlinearity.get_true_key()
        self.nl = (
            nn.ReLU(inplace=True) if self.nonlinearity == "relu" else nn.Tanh()
        )

        # Encoder
        conv_settings, pool_settings = self.params.cnn_encoder.conv, self.params.cnn_encoder.pool
        conv_params = conv_settings.kernel_size, conv_settings.stride, conv_settings.padding
        pool_params = pool_settings.kernel_size, pool_settings.stride, pool_settings.padding
        self.convs = model_utils.get_conv(
            self.input_channels, self.output_channels, self.hidden_channels,
            conv_params, pool_params, nonlinearity=self.nonlinearity
        )

        # MLP
        output_width, output_height = model_utils.conv_block_output_size(
            self.convs, self.input_width, self.input_height
        )
        assert output_width > 0 and output_height > 0
        self.mlp = model_utils.get_linear(
            output_width * output_height * self.output_channels,
            self.mlp_output_size, self.mlp_hidden_sizes, nonlinearity=self.nonlinearity
        )

        # GNN
        self.gnn_conv = nn.ModuleList()
        sizes = (
            [self.mlp_output_size] +
            self.gnn_hidden_sizes +
            [self.embedding_size]
        )
        for i in range(1, len(sizes)):
            self.gnn_conv.append(
                gnn.GATConv(
                    sizes[i - 1], sizes[i], add_self_loops=False,
                    heads=2, concat=False
                )
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
        embeddings = None
        for conv in self.gnn_conv:
            features = conv(features, states.edge_index)
            embeddings = features
            features = self.nl(features)
            features = F.dropout(
                features, p=self.dropout, training=self.training
            )

        return embeddings
