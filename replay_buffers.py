import random
from collections import namedtuple, deque

import torch
import numpy as np

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
