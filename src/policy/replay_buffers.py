import random
import pickle
from collections import namedtuple, deque

import numpy as np
import torch
from torch_geometric.data import Data, Batch


Experience = namedtuple(
    "Experience", field_names=[
        "state", "choice", "reward", "next_state",
        "next_legal_choices", "finished"
    ]
)
MultiExperience = namedtuple(
    "Experience", field_names=[
        "state", "choice", "adjacency", "reward", "next_state",
        "next_legal_choices", "next_adjacency", "finished", "inactive"
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

    def add(self, experience, multi=False):
        '''
        Add a new experience to memory
        '''
        if not multi:
            self.memory.append(Experience(*experience))
        else:
            self.memory.append(MultiExperience(*experience))

    def sample(self, multi=False):
        '''
        Randomly sample a batch of experiences from memory.
        Each returned tensor has shape (batch_size, *)
        '''
        if not multi:
            states, choices, rewards, next_states, next_legal_choices, finished = zip(
                *random.sample(self.memory, k=self.batch_size)
            )
        else:
            states, choices, adjacencies, rewards, next_states, next_legal_choices, next_adjacencies, finished, inactives = zip(
                *random.sample(self.memory, k=self.batch_size)
            )

        # Check for PyTorch Geometric
        if isinstance(states[0], np.ndarray):
            states = torch.tensor(
                states, dtype=torch.float32, device=self.device
            )
            next_states = torch.tensor(
                next_states, dtype=torch.float32, device=self.device
            )
        elif isinstance(states[0], Data):
            states = Batch.from_data_list(states).to(self.device)
            next_states = Batch.from_data_list(next_states).to(self.device)

        choices = torch.tensor(
            choices, dtype=torch.int64, device=self.device
        ).reshape((self.batch_size, -1))
        rewards = torch.tensor(
            rewards, dtype=torch.float32, device=self.device
        ).reshape((self.batch_size, -1))
        next_legal_choices = torch.tensor(
            next_legal_choices, dtype=torch.bool, device=self.device
        ).reshape((self.batch_size, -1))
        finished = torch.tensor(
            finished, dtype=torch.uint8, device=self.device
        ).reshape((self.batch_size, -1))

        if multi:
            adjacencies = torch.tensor(
                adjacencies, dtype=torch.int64, device=self.device
            ).reshape((self.batch_size, -1))
            next_adjacencies = torch.tensor(
                next_adjacencies, dtype=torch.int64, device=self.device
            ).reshape((self.batch_size, -1))
            inactives = torch.tensor(
                inactives, dtype=torch.uint8, device=self.device
            ).reshape((self.batch_size, -1))
            return (
                states, choices, adjacencies, rewards,
                next_states, next_legal_choices, next_adjacencies,
                finished, inactives
            )

        return states, choices, rewards, next_states, next_legal_choices, finished

    def can_sample(self):
        '''
        Check if there are enough samples in the replay buffer
        '''
        return len(self.memory) >= self.batch_size

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
