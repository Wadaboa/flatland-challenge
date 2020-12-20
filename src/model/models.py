import numpy as np
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F

import model.model_utils as model_utils


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


def conv_bn_act(input_channels, output_channels, kernel_size=3,
                stride=1, padding=0, nonlinearity="relu"):
    '''
    Returns a block composed by a convolutional layer and a batch norm one,
    followed by a non-linearity (e.g. ReLU or Tanh)
    '''
    return [
        nn.Conv2d(
            input_channels, output_channels,
            kernel_size=kernel_size, stride=stride, padding=padding
        ),
        nn.BatchNorm2d(output_channels),
        nn.ReLU() if nonlinearity == "relu" else nn.Tanh()
    ]


def conv_bn_act_maxpool(input_channels, output_channels, kernel_size=3,
                        stride=1, padding=0, nonlinearity="relu"):
    '''
    Returns a block composed by a convolutional layer and a batch norm one,
    followed by a non-linearity (e.g. ReLU or Tanh), with a final max pooling layer
    '''
    return (
        conv_bn_act(
            input_channels, output_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, nonlinearity=nonlinearity
        ) +
        [nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)]
    )


def get_conv(input_channels, output_channels, hidden_channels,
             kernel_size=3, stride=1, padding=0, nonlinearity="relu", pool=False):
    '''
    Returns a PyTorch Sequential object containing `conv_bn_act` or `conv_bn_act_pool` blocks,
    by following the given input/hidden/output number of channels
    '''
    assert len(hidden_channels) >= 1
    conv_block = conv_bn_act if not pool else conv_bn_act_maxpool
    conv = conv_block(
        input_channels, hidden_channels[0], kernel_size=kernel_size,
        stride=stride, padding=padding, nonlinearity=nonlinearity
    )
    for i in range(1, len(hidden_channels)):
        conv.extend(
            conv_block(
                hidden_channels[i - 1], hidden_channels[i],
                kernel_size=kernel_size, stride=stride, padding=padding,
                nonlinearity=nonlinearity
            )
        )
    conv.extend(
        conv_block(
            hidden_channels[-1], output_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, nonlinearity=nonlinearity
        )
    )
    return nn.Sequential(*conv)


######################################################################
################################# DQN ################################
######################################################################

class DQN(nn.Module):
    '''
    Vanilla deep Q-Network
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
##################### Single agent DQN + GNN #########################
######################################################################

class SingleDQNGNN(DQN):
    '''
    Single agent DQN + GNN
    '''

    def __init__(self, state_size, action_size, pos_size, embedding_size,
                 hidden_sizes=[128, 128], nonlinearity="tanh",
                 gnn_hidden_size=16, depth=3, dropout=0.0):
        super(SingleDQNGNN, self).__init__(
            action_size * embedding_size, action_size,
            hidden_sizes=hidden_sizes, nonlinearity=nonlinearity
        )
        self.pos_size = pos_size
        self.embedding_size = embedding_size
        self.depth = depth
        self.dropout = dropout
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
            ), dtype=torch.float
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
                        [-self.depth] * self.embedding_size, dtype=torch.float
                    )
                else:
                    embs[i, j] = emb[p.item()]

        # Call the DQN with a tensor of shape
        # (batch_size, pos_size, embedding_size)
        return super().forward(embs)


######################################################################
###################### Multi agent DQN + GNN #########################
######################################################################

class MultiDQNGNN(DQN):
    '''
    Multi agent DQN + GNN
    '''

    def __init__(self, action_size, input_width, input_height, input_channels, output_channels,
                 hidden_channels=[16, 32, 16], embedding_size=128, hidden_sizes=[128, 128],
                 nonlinearity="relu"):
        super(MultiDQNGNN, self).__init__(
            embedding_size, action_size,
            hidden_sizes=hidden_sizes, nonlinearity=nonlinearity
        )
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels
        self.embedding_size = embedding_size

        # Encoder
        self.convs = get_conv(
            input_channels, output_channels, hidden_channels,
            kernel_size=3, stride=1, padding=0,
            nonlinearity=nonlinearity, pool=False
        )

        # MLP
        output_width, output_height = model_utils.conv_block_output_size(
            self.convs, input_width, input_height
        )
        self.mlp = get_linear(
            output_width * output_height * output_channels,
            embedding_size, hidden_sizes, nonlinearity=nonlinearity
        )

        # GNN
        self.gnn_conv = gnn.GCNConv(embedding_size, embedding_size)

    def forward(self, state, adjacency, inactives):
        q_values = torch.zeros(
            (state.shape[0], state.shape[1], self.action_size),
            dtype=torch.float
        )
        for batch_number, batch in enumerate(state):
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
            num_agents = adjacency.shape[0]
            for i in range(num_agents):
                for j in range(num_agents):
                    if adjacency[i, j] != 0:
                        edge_index.append([i, j])
                        edge_weight.append(adjacency[i, j])
            edge_index = torch.tensor(
                edge_index, dtype=torch.long
            ).t().contiguous()
            edge_weight = torch.tensor(edge_weight, dtype=torch.float)

            # Compute embeddings for each node by performing graph convolutions
            embeddings = self.gnn_conv(
                features, edge_index, edge_weight=edge_weight
            )

            # Call the DQN with the embeddings associated to active agents
            for handle in torch.nonzero(~inactives, as_tuple=True)[0]:
                q_values[batch_number, handle.item(), :] = (
                    super().forward(embeddings[handle.item()].unsqueeze(0))
                )

        # Return the Q-values tensor
        return q_values
