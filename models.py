from collections import namedtuple, deque, Iterable
import os
import copy
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count
from PIL import Image
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.modules.activation import MultiheadAttention


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


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


class Input(nn.Module):

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


######################################################################
############################## DQN ###################################
######################################################################

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 512  # minibatch size
GAMMA = 0.99  # discount factor 0.99
TAU = 1e-3  # for soft update of target parameters
LR = 0.5e-4  # learning rate 0.5e-4 works
UPDATE_EVERY = 10  # how often to update the network


class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidsize1=128, hidsize2=128):
        super(DuelingQNetwork, self).__init__()

        self.fc1_val = nn.Linear(state_size, hidsize1)
        self.fc2_val = nn.Linear(hidsize1, hidsize2)
        self.fc3_val = nn.Linear(hidsize2, 1)

        self.fc1_adv = nn.Linear(state_size, hidsize1)
        self.fc2_adv = nn.Linear(hidsize1, hidsize2)
        self.fc3_adv = nn.Linear(hidsize2, action_size)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        val = F.relu(self.fc1_val(x))
        val = F.relu(self.fc2_val(val))
        val = self.fc3_val(val)

        # advantage calculation
        adv = F.relu(self.fc1_adv(x))
        adv = F.relu(self.fc2_adv(adv))
        adv = self.fc3_adv(adv)
        return val + adv - adv.mean()


class Policy:
    def step(self, state, action, reward, next_state, done):
        raise NotImplementedError

    def act(self, state, eps=0.):
        raise NotImplementedError

    def save(self, filename):
        raise NotImplementedError

    def load(self, filename):
        raise NotImplementedError


class DDDQNPolicy(Policy):
    """Dueling Double DQN policy"""

    def __init__(self, state_size, action_size, parameters, evaluation_mode=False):
        self.evaluation_mode = evaluation_mode

        self.state_size = state_size
        self.action_size = action_size
        self.double_dqn = True
        self.hidsize = 1

        if not evaluation_mode:
            self.hidsize = parameters.hidden_size
            self.buffer_size = parameters.buffer_size
            self.batch_size = parameters.batch_size
            self.update_every = parameters.update_every
            self.learning_rate = parameters.learning_rate
            self.tau = parameters.tau
            self.gamma = parameters.gamma
            self.buffer_min_size = parameters.buffer_min_size
            self.loss = torch.tensor(0.0)
            self.time_step = 0

        # Device
        if parameters.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("🐇 Using GPU")
        else:
            self.device = torch.device("cpu")
            print("🐢 Using CPU")

        # Q-Network
        self.qnetwork_local = DuelingQNetwork(
            state_size, action_size, hidsize1=self.hidsize, hidsize2=self.hidsize).to(self.device)

        if not evaluation_mode:
            self.qnetwork_target = copy.deepcopy(self.qnetwork_local)
            self.optimizer = optim.Adam(
                self.qnetwork_local.parameters(), lr=self.learning_rate)
            self.memory = ReplayBuffer(
                action_size, self.batch_size, self.buffer_size, self.device)

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def step(self, state, action, reward, next_state, done):
        assert not self.evaluation_mode, "Policy has been initialized for evaluation only."

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every `update_every` time steps
        self.time_step = (self.time_step + 1) % self.update_every
        if self.time_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.buffer_min_size and len(self.memory) >= self.batch_size:
                self._learn()

    def _learn(self):
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences

        # Get expected Q values from local model
        q_expected = self.qnetwork_local(
            states
        ).gather(1, actions.unsqueeze(-1)).squeeze()

        if self.double_dqn:
            # Take the maximum probabilities of actions for each sample in the mini-batch
            # and return a matrix of shape (1, batch-size), where
            # each element represents the best action itself
            q_best_action = self.qnetwork_local(next_states).detach().max(1)[1]

            # Get expected Q values from target model
            q_targets_next = self.qnetwork_target(
                next_states
            ).detach().gather(1, q_best_action.unsqueeze(-1)).squeeze()
        else:
            q_targets_next = self.qnetwork_target(
                next_states
            ).detach().max(1)[0]

        # Compute Q targets for current states
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

        # Compute loss
        self.loss = F.mse_loss(q_expected, q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        # Update target network
        self._soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def _soft_update(self, local_model, target_model, tau):
        # Soft update model parameters
        # θ_target = τ * θ_local + (1 - τ) * θ_target
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )

    def save(self, filename):
        torch.save(self.qnetwork_local.state_dict(), filename + ".local")
        torch.save(self.qnetwork_target.state_dict(), filename + ".target")

    def load(self, filename):
        if os.path.exists(filename + ".local"):
            self.qnetwork_local.load_state_dict(
                torch.load(filename + ".local")
            )
        if os.path.exists(filename + ".target"):
            self.qnetwork_target.load_state_dict(
                torch.load(filename + ".target")
            )

    def save_replay_buffer(self, filename):
        memory = self.memory.memory
        with open(filename, 'wb') as f:
            pickle.dump(list(memory)[-500000:], f)

    def load_replay_buffer(self, filename):
        with open(filename, 'rb') as f:
            self.memory.memory = pickle.load(f)

    def test(self):
        self.act(np.array([[0] * self.state_size]))
        self._learn()


Experience = namedtuple(
    "Experience", field_names=[
        "state", "action", "reward", "next_state", "done"
    ]
)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, batch_size, buffer_size, device):
        '''
        Initialize a ReplayBuffer object.
        '''
        self.action_size = action_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        '''
        Add a new experience to memory
        '''
        self.memory.append(
            Experience(state, action, reward, next_state, done)
        )

    def sample(self):
        '''
        Randomly sample a batch of experiences from memory
        '''
        states, actions, rewards, next_states, dones = zip(
            *random.sample(self.memory, k=self.batch_size)
        )
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.tensor(
            rewards, dtype=torch.float32, device=self.device
        )
        next_states = torch.tensor(
            next_states, dtype=torch.float32, device=self.device
        )
        dones = torch.tensor(dones, dtype=torch.uint8, device=self.device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        '''
        Return the current size of internal memory
        '''
        return len(self.memory)
