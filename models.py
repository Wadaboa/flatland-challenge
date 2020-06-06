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


class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)


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


######build the model#########
neighbors = 4
action_space = 4
n_data = 20
encoder = MLP()
m1, m1_r = MultiheadAttention(embed_dim=128, num_heads=neighbors)
m2, m2_r = MultiheadAttention(embed_dim=128, num_heads=neighbors)
q_net = QNet(action_dim=action_space)
vec = np.zeros((1, neighbors))
vec[0][0] = 1

In = []
for j in range(n_data):
    In.append(Input(shape=[len_feature]))
    In.append(Input(shape=(neighbors, n_data)))
In.append(Input(shape=(1, neighbors)))
feature = []
for j in range(n_data):
    feature.append(encoder(In[j*2]))

feature_ = Concatenate(axis=1)(feature)

relation1 = []
for j in range(n_data):
    T = Lambda(lambda x: K.batch_dot(x[0], x[1]))([In[j*2+1], feature_])
    relation1.append(m1([T, T, T, In[n_data*2]]))

relation1_ = Concatenate(axis=1)(relation1)

relation2 = []
for j in range(n_data):
    T = Lambda(lambda x: K.batch_dot(x[0], x[1]))([In[j*2+1], relation1_])
    relation2.append(m2([T, T, T, In[n_data*2]]))

V = []
for j in range(n_data):
    V.append(q_net([feature[j], relation1[j], relation2[j]]))

model = Model(input=In, output=V)
model.compile(optimizer=Adam(lr=0.0001), loss='mse')
