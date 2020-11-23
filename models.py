import numpy as np

import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F
from torch.nn.functional import softmax


def get_seq(input_size, output_size, hidden_sizes, module=nn.Linear, nonlinearity="tanh"):
    '''
    Returns a PyTorch Sequential object containing `module` typed layers with
    non-linear activation functions, by following the given input/hidden/output sizes
    '''
    assert len(hidden_sizes) >= 1
    nl = nn.ReLU() if nonlinearity == "relu" else nn.Tanh()
    fc = [module(input_size, hidden_sizes[0]), nl]
    for i in range(1, len(hidden_sizes)):
        fc.extend([module(hidden_sizes[i - 1], hidden_sizes[i]), nl])
    fc.extend([module(hidden_sizes[-1], output_size)])
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
        self.fc = get_seq(
            state_size, action_size, hidden_sizes,
            module=nn.Linear, nonlinearity=nonlinearity
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
        self.aggregation = aggregation
        self.fc_val = get_seq(
            state_size, 1, hidden_sizes,
            module=nn.Linear, nonlinearity=nonlinearity
        )
        self.fc_adv = get_seq(
            state_size, action_size, hidden_sizes,
            module=nn.Linear, nonlinearity=nonlinearity
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

    def __init__(self, state_size, action_size, hidden_sizes=[128, 128], nonlinearity="relu",
                 gnn_hidden_size=16, embedding_size=100):
        super(DQNGNN, self).__init__(
            state_size, action_size, hidden_sizes=hidden_sizes, nonlinearity=nonlinearity
        )
        '''
        self.gnn_conv = get_seq(
            state_size, embedding_size, gnn_hidden_sizes,
            module=gnn.GCNConv, nonlinearity=nonlinearity
        )
        '''
        self.conv1 = gnn.GCNConv(state_size, gnn_hidden_size)
        self.conv2 = gnn.GCNConv(gnn_hidden_size, gnn_hidden_size)
        self.conv3 = gnn.GCNConv(gnn_hidden_size, embedding_size)

    def forward(self, state):
        x, edge_index = state.x, state.edge_index
        x = self.conv1(x, edge_index)
        x = nn.ReLU(x)
        x = self.conv2(x, edge_index)
        x = nn.ReLU(x)
        x = self.conv3(x, edge_index)
        x = nn.ReLU(x)
        return self.fc(x)
