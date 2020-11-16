import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softmax


def get_linear(input_size, output_size, hidden_sizes):
    '''
    Returns a PyTorch Sequential object containing FC layers with
    ReLU activation functions, by following the given input/hidden/output sizes
    '''
    assert len(hidden_sizes) >= 1
    fc = [nn.Linear(input_size, hidden_sizes[0]), nn.ReLU()]
    for i in enumerate(1, len(hidden_sizes)):
        fc.extend([
            nn.Linear(hidden_sizes[i - 1], hidden_sizes[i])
            nn.ReLU()
        ])
    fc.extend([nn.Linear(hidden_sizes[-1], output_size)])
    return nn.Sequential(fc)


######################################################################
################################# DQN ################################
######################################################################


class DQN(nn.Module):
    '''
    Deep Q-Network
    '''

    def __init__(self, state_size, action_size, hidden_sizes=[128, 128]):
        super(DQN, self).__init__()
        self.fc = get_linear(state_size, action_size, hidden_sizes)

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

    def __init__(self, state_size, action_size, hidden_sizes=[128, 128], aggregation="mean"):
        super(DuelingDQN, self).__init__()
        self.aggregation = aggregation
        self.fc_val = get_linear(state_size, 1, hidden_sizes)
        self.fc_adv = get_linear(state_size, action_size, hidden_sizes)

    def forward(self, state):
        state = torch.flatten(state, start_dim=1)
        val = self.fc_val(state)
        adv = self.fc_adv(state)
        agg = adv.mean() if self.aggregation == "mean" else adv.max()
        return val + adv - agg
