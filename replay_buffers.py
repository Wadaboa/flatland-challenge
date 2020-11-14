import random
import pickle

from collections import namedtuple, deque

import torch
import numpy as np
from torch._C import dtype

Experience = namedtuple(
    "Experience", field_names=[
        "state", "legal_choices", "action",
        "reward", "next_state", "next_legal_choices", "done"
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

    def add(self, state, legal_choices, action, reward, next_state, next_legal_choices, done):
        '''
        Add a new experience to memory
        '''
        self.memory.append(
            Experience(state, legal_choices, action, reward, next_state,
                       next_legal_choices, done)
        )

    def sample(self):
        '''
        Randomly sample a batch of experiences from memory
        '''
        states, legal_choices, actions, rewards, next_states, next_legal_choices, dones = zip(
            *random.sample(self.memory, k=self.batch_size)
        )
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        legal_choices = torch.tensor(
            legal_choices, dtype=torch.bool, device=self.device
        )
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.tensor(
            rewards, dtype=torch.float32, device=self.device
        )
        next_states = torch.tensor(
            next_states, dtype=torch.float32, device=self.device
        )
        next_legal_choices = torch.tensor(
            next_legal_choices, dtype=torch.bool, device=self.device
        )
        dones = torch.tensor(dones, dtype=torch.uint8, device=self.device)
        return states, legal_choices, actions, rewards, next_states, next_legal_choices, dones

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(list(self.memory), f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.memory = pickle.load(f)

    def __len__(self):
        '''
        Return the current size of internal memory
        '''
        return len(self.memory)
