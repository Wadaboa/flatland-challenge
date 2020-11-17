import os
import copy
import random
import pickle

import numpy as np
import torch
import torch.optim as optim

from models import DQN, DuelingDQN
from replay_buffers import ReplayBuffer

import env_utils
import model_utils
import action_selection


class Policy:
    '''
    Policy abstract class
    '''

    def __init__(self, state_size=None, choice_size=None):
        self.state_size = state_size
        self.choice_size = choice_size

    def act(self, state, legal_choices=None):
        raise NotImplementedError()

    def step(self, experience):
        raise NotImplementedError()

    def save(self, filename):
        raise NotImplementedError()

    def load(self, filename):
        raise NotImplementedError()


class RandomPolicy(Policy):
    '''
    Policy which chooses random moves
    '''

    def __init__(self, state_size=None, choice_size=None):
        self.action_selector = action_selection.RandomActionSelector()
        super(RandomPolicy, self).__init__(state_size, choice_size)

    def act(self, state, legal_choices=None):
        choices = np.array(env_utils.RailEnvChoices.values())
        return self.action_selector.select(choices, legal_choices)

    def step(self, experience):
        return None

    def save(self, filename):
        return None

    def load(self, filename):
        return None


class DQNPolicy(Policy):
    '''
    DQN abstract policy
    '''

    PARAMETERS = {
        "buffer_size": int(1e5),
        "min_buffer_size": 0,
        "batch_size": 128,
        "update_every": 8,
        "learning_rate": 0.5e-4,
        "tau": 1e-3,
        "discount": 0.99,
        "hidden_sizes": [128, 128],
        "dueling": True,
        "double": True,
        "softmax_bellman": True
        "loss": "huber"
    }

    def __init__(self, state_size, choice_size, choice_selector, training=False):
        '''
        Initialize DQNPolicy object
        '''
        super(DQNPolicy, self).__init__(state_size, choice_size)
        assert (
            isinstance(choice_selector, ActionSelector),
            "The choice selection object must be an instance of ActionSelector"
        )

        # Parameters
        self.state_size = state_size
        self.choice_size = choice_size
        self.choice_selector = choice_selector
        self.training = training
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("🐇 Using GPU")

        # Q-Network
        net = DuelingDQN if self.PARAMETERS["dueling"] else DQN
        self.qnetwork_local = net(
            self.state_size, self.choice_size, hidden_sizes=self.PARAMETERS["hidden_sizes"]
        ).to(self.device)

        # Training parameters
        if training:
            self.time_step = 0
            self.qnetwork_target = copy.deepcopy(self.qnetwork_local)
            self.optimizer = optim.Adam(
                self.qnetwork_local.parameters(), lr=self.learning_rate
            )
            self.criterion = (
                nn.SmoothL1Loss() if self.PARAMETERS["dueling"] == "huber"
                else nn.MSELoss()
            )
            self.loss = torch.tensor(0.0)
            self.memory = ReplayBuffer(
                self.choice_size, self.batch_size, self.buffer_size, self.device
            )

    def act(self, state, legal_choices):
        '''
        Perform action selection based on the Q-values returned by the network
        '''
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            choice_values = self.qnetwork_local(state)
        return (
            self.choice_selector.select(choice_values, legal_choices) if training
            else np.argmax(choice_values.detach().cpu().numpy())
        )

    def step(self, experience):
        '''
        Add an experience to memory and eventually perform a training step
        '''
        assert (
            self.training,
            "Policy has been initialized for evaluation only"
        )

        # Save experience in replay memory
        self.memory.add(experience)

        # Learn every `update_every` time steps
        self.time_step = (self.time_step + 1) % self.update_every
        if self.time_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.buffer_min_size and len(self.memory) >= self.batch_size:
                self.qnetwork_local.train()
                self._learn()

    def _learn(self):
        '''
        Perform a learning step
        '''
        # Sample a batch of experiences
        experiences = self.memory.sample()
        states, legal_choices, choices, rewards, next_states, next_legal_choices, dones = experiences

        # Get expected Q-values from local model
        q_expected = self.qnetwork_local(states).gather(1, choices)

        # Get expected Q-values from target model
        q_targets_next = torch.from_numpy(
            self._get_q_targets_next(next_legal_choices)
        ).to(self.device)

        # Compute Q-targets for current states
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

        # Compute and minimize the loss
        self.loss = self.criterion(q_expected, q_targets)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        # Update target network
        self._soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def _get_q_targets_next(self, next_legal_choices):
        '''
        Get expected Q-values from target network
        '''

        def _double_dqn():
            q_locals_next = self.qnetwork_local(
                next_states
            ).detach().cpu().numpy()
            q_targets_next = self.qnetwork_target(
                next_states
            ).detach().cpu().numpy()

            # Softmax Bellman
            if self.PARAMETERS["softmax_bellman"]:
                return np.sum(
                    q_targets_next * model_utils.masked_softmax(
                        q_targets_next, next_legal_choices
                    ), axis=1, keepdims=True
                )

            # Standard Bellman
            best_choices = model_utils.masked_argmax(
                q_targets_next, next_legal_choices
            )
            return q_targets_next[best_choices]

        def _dqn():
            q_targets_next = self.qnetwork_target(
                next_states
            ).detach().cpu().numpy()

            # Standard or softmax Bellman
            return (
                model_utils.masked_max(q_targets_next, next_legal_choices) if not self.PARAMETERS["softmax_bellman"]
                else np.sum(model_utils.masked_softmax(q_targets_next, next_legal_choices) * q_targets_next, axis=1, keepdims=True)
            )

        return _double_dqn() if self.PARAMETERS["double"] else _dqn()

    def _soft_update(self, local_model, target_model, tau):
        '''
        Soft update model parameters: θ_target = τ * θ_local + (1 - τ) * θ_target
        '''
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )

    def save(self, filename):
        '''
        Save both local and targets networks parameters
        '''
        torch.save(self.qnetwork_local.state_dict(), filename + ".local")
        torch.save(self.qnetwork_target.state_dict(), filename + ".target")

    def load(self, filename):
        '''
        Load only the local network if evaluating,
        otherwise load both local and target networks
        '''
        if os.path.exists(filename + ".local"):
            self.qnetwork_local.load_state_dict(
                torch.load(filename + ".local")
            )
            if self.training and os.path.exists(filename + ".target"):
                self.qnetwork_target.load_state_dict(
                    torch.load(filename + ".target")
                )

    def save_replay_buffer(self, filename):
        '''
        Save the current replay buffer
        '''
        self.memory.save(filename)

    def load_replay_buffer(self, filename):
        '''
        Load a stored representation of the replay buffer
        '''
        self.memory.load(filename)
