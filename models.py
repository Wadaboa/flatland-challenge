import math
import random
import numpy as np
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.nn.modules.activation import MultiheadAttention


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward')
)


class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


######################################################################
############################## DGN ###################################
######################################################################

class MLP(nn.Module):

    def __init__(self, features_lenght):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(features_lenght, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.view(-1, 1, 128)
        return x


class Input(nn.module):

    def __init__(self, shape):
        super(Input, self).__init__()
        self.shape = shape

    def forward(self, inp):
        if inp.size() == self.shape:
            return inp
        return None


class Flatten(nn.Module):

    def forward(self, inp):
        return inp.view(inp.size(0), -1)


class Lambda(nn.Module):

    def __init__(self, func):
        super(Lambda, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class QNet(nn.Module):

    def __init__(self, action_dim):
        super(QNet, self).__init__()
        self.flatten_layer = nn.Flatten()
        self.dense_layer = nn.Linear(384, action_dim)

    def forward(self, i1, i2, i3):
        x1 = self.flatten_layer(i1)
        x2 = self.flatten_layer(i2)
        x3 = self.flatten_layer(i3)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.dense_layer(x)
        return x


def batch_dot(a, b):
    B = a.shape[0]
    S = a.shape[1]
    return torch.bmm(a.view(B, 1, S), b.view(B, S, 1)).reshape(-1)
