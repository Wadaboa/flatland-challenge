import random
import pickle

from collections import namedtuple, deque

import numpy as np
import torch


Experience = namedtuple(
    "Experience", field_names=[
        "state", "choice",
        "reward", "next_state", "next_legal_choices", "done"
    ]
)


class ReplayBuffer:
    '''
    Fixed-size buffer to store experience tuples
    '''

    def __init__(self, choice_size, batch_size, buffer_size, device):
        '''
        Initialize a ReplayBuffer object
        '''
        self.choice_size = choice_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
        self.device = device

    def add(self, experience):
        '''
        Add a new experience to memory
        '''
        state, choice, reward, next_state, next_legal_choices, done = experience
        self.memory.append(
            Experience(
                state, choice, reward,
                next_state, next_legal_choices, done
            )
        )

    def sample(self):
        '''
        Randomly sample a batch of experiences from memory.
        Each returned tensor has shape (batch_size, *)
        '''
        states, choices, rewards, next_states, next_legal_choices, dones = zip(
            *random.sample(self.memory, k=self.batch_size)
        )
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        choices = torch.tensor(
            choices, dtype=torch.int64, device=self.device
        ).reshape((self.batch_size, -1))
        rewards = torch.tensor(
            rewards, dtype=torch.float32, device=self.device
        ).reshape((self.batch_size, -1))
        next_states = torch.tensor(
            next_states, dtype=torch.float32, device=self.device
        )
        next_legal_choices = torch.tensor(
            next_legal_choices, dtype=torch.bool, device=self.device
        )
        dones = torch.tensor(
            dones, dtype=torch.uint8, device=self.device
        ).reshape((self.batch_size, -1))
        return states, choices, rewards, next_states, next_legal_choices, dones

    def save(self, filename):
        '''
        Save the current replay buffer to a pickle file
        '''
        with open(filename, 'wb') as f:
            pickle.dump(list(self.memory), f)

    def load(self, filename):
        '''
        Load the current replay buffer from the given pickle file
        '''
        with open(filename, 'rb') as f:
            self.memory = pickle.load(f)

    def __len__(self):
        '''
        Return the current size of internal memory
        '''
        return len(self.memory)
