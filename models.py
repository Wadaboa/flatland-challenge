import numpy as np

import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F
from torch.nn.functional import softmax


def get_linear(input_size, output_size, hidden_sizes, nonlinearity="tanh"):
    '''
    Returns a PyTorch Sequential object containing FC layers with
    non-linear activation functions, by following the given input/hidden/output sizes
    '''
    assert len(hidden_sizes) >= 1
    nl = nn.ReLU() if nonlinearity == "relu" else nn.Tanh()
    fc = [nn.Linear(input_size, hidden_sizes[0]), nl]
    for i in range(1, len(hidden_sizes)):
        fc.extend([nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]), nl])
    fc.extend([nn.Linear(hidden_sizes[-1], output_size)])
    return nn.Sequential(*fc)


######################################################################
################################# DQN ################################
######################################################################


class DQN(nn.Module):
    '''
    Deep Q-Network
    '''

    def __init__(self, state_size, action_size, hidden_sizes=[128, 128], nonlinearity="tanh"):
        super(DQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.fc = get_linear(
            state_size, action_size, hidden_sizes, nonlinearity=nonlinearity
        )

    def forward(self, state):
        state = torch.flatten(state, start_dim=1)
        return self.fc(state)


######################################################################
########################### Dueling DQN ##############################
######################################################################

class DuelingDQN(nn.Module):
    '''
    Dueling DQN
    '''

    def __init__(self, state_size, action_size, hidden_sizes=[128, 128], nonlinearity="tanh", aggregation="mean"):
        super(DuelingDQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.aggregation = aggregation
        self.fc_val = get_linear(
            state_size, 1, hidden_sizes, nonlinearity=nonlinearity
        )
        self.fc_adv = get_linear(
            state_size, action_size, hidden_sizes, nonlinearity=nonlinearity
        )

    def forward(self, state):
        state = torch.flatten(state, start_dim=1)
        val = self.fc_val(state)
        adv = self.fc_adv(state)
        agg = adv.mean() if self.aggregation == "mean" else adv.max()
        return val + adv - agg


######################################################################
############################# DQN + GNN ##############################
######################################################################

class DQNGNN(DQN):
    '''
    DQN + GNN
    '''

    def __init__(self, state_size, action_size, embedding_size,
                 hidden_sizes=[128, 128], nonlinearity="tanh",
                 gnn_hidden_size=16, depth=3, dropout=0.0):
        super(DQNGNN, self).__init__(
            action_size * embedding_size, action_size,
            hidden_sizes=hidden_sizes, nonlinearity=nonlinearity
        )
        self.embedding_size = embedding_size
        self.depth = depth
        self.dropout = dropout
        self.nl = nn.ReLU() if nonlinearity == "relu" else nn.Tanh()
        self.gnn_conv = nn.ModuleList()
        self.gnn_conv.append(gnn.GCNConv(state_size, gnn_hidden_size))
        for i in range(1, self.depth - 1):
            self.gnn_conv.append(gnn.GCNConv(gnn_hidden_size, gnn_hidden_size))
        self.gnn_conv.append(gnn.GCNConv(gnn_hidden_size, self.embedding_size))

    def forward(self, state):
        embs = torch.empty(
            size=(
                len(state),
                self.action_size,
                self.embedding_size
            ), dtype=torch.float
        )
        for i, graph in enumerate(state):
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
                        [-1] * self.embedding_size, dtype=torch.float
                    )
                else:
                    embs[i, j] = emb[p.item()]

        # Call the DQN with a tensor of shape
        # (batch_size, action_size, embedding_size)
        return super().forward(embs)
