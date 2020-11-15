import os
import copy
import random
import pickle

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from flatland.envs.rail_env import RailEnvActions
from models import DDDQNetwork
from replay_buffers import ReplayBuffer


class Policy:

    def __init__(self, state_size=None, action_size=None):
        self.state_size = state_size
        self.action_size = action_size

    def act(self, state, *args):
        raise NotImplementedError()

    def step(self, memories):
        raise NotImplementedError()

    def save(self, filename):
        raise NotImplementedError()

    def load(self, filename):
        raise NotImplementedError()


class RandomPolicy(Policy):

    def __init__(self, state_size=None, action_size=None):
        super(RandomPolicy, self).__init__(state_size, action_size)

    def act(self, state):
        return np.random.choice([
            RailEnvActions.MOVE_FORWARD,
            RailEnvActions.MOVE_LEFT,
            RailEnvActions.MOVE_RIGHT
        ])

    def step(self, memories):
        return None

    def save(self, filename):
        return None

    def load(self, filename):
        return None


class ShortestPathPolicy(Policy):

    def __init__(self, state_size=None, action_size=None):
        super(RandomPolicy, self).__init__(state_size, action_size)

    def act(self, state):
        print(state)
        return state[4]

    def step(self, memories):
        return None

    def save(self, filename):
        return None

    def load(self, filename):
        return None


class DDDQNPolicy(Policy):
    """Dueling Double DQN policy"""

    def __init__(self, state_size, action_size, evaluation_mode=False, parameters=None):
        self.evaluation_mode = evaluation_mode

        self.state_size = state_size
        self.action_size = action_size
        self.double_dqn = True
        self.hidsize = 128

        self.device = torch.device("cpu")

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

        # Q-Network
        self.qnetwork_local = DDDQNetwork(
            state_size, action_size, hidsize1=self.hidsize, hidsize2=self.hidsize).to(self.device)

        if not evaluation_mode:
            self.qnetwork_target = copy.deepcopy(self.qnetwork_local)
            self.optimizer = optim.Adam(
                self.qnetwork_local.parameters(), lr=self.learning_rate)
            self.memory = ReplayBuffer(
                action_size, self.batch_size, self.buffer_size, self.device)

    def act(self, state, legal_choices, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        legal_choices = torch.tensor(
            legal_choices
        ).bool().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            choice_values = self.qnetwork_local(state, legal_choices)

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(choice_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def step(self, state, legal_choices, action, reward, next_state, next_legal_choices, done):
        assert not self.evaluation_mode, "Policy has been initialized for evaluation only."

        # Save experience in replay memory
        self.memory.add(state, legal_choices, action, reward,
                        next_state, next_legal_choices, done)

        # Learn every `update_every` time steps
        self.time_step = (self.time_step + 1) % self.update_every
        if self.time_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.buffer_min_size and len(self.memory) >= self.batch_size:
                self.qnetwork_local.train()
                self._learn()

    def _learn(self):
        experiences = self.memory.sample()
        states, legal_choices, actions, rewards, next_states, next_legal_choices, dones = experiences

        # Get expected Q values from local model
        q_expected = self.qnetwork_local(
            states, legal_choices
        ).gather(1, actions.unsqueeze(-1)).squeeze()

        if self.double_dqn:
            # Take the maximum probabilities of actions for each sample in the mini-batch
            # and return a matrix of shape (1, batch-size), where
            # each element represents the best action itself
            q_best_action = self.qnetwork_local(
                next_states, next_legal_choices).detach().max(1)[1]

            # Get expected Q values from target model
            q_targets_next = self.qnetwork_target(
                next_states, next_legal_choices
            ).detach().gather(1, q_best_action.unsqueeze(-1)).squeeze()
        else:
            q_targets_next = self.qnetwork_target(
                next_states, next_legal_choices
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
        if not self.evaluation_mode and os.path.exists(filename + ".target"):
            self.qnetwork_target.load_state_dict(
                torch.load(filename + ".target")
            )

    def save_replay_buffer(self, filename):
        self.memory.save(filename)

    def load_replay_buffer(self, filename):
        self.memory.load(filename)

    def test(self):
        self.act(np.array([[0] * self.state_size]))
        self._learn()


class ActionSelection:

    def select(actions, legal_actions):
        pass


class EpsilonGreedyActionSelection(ActionSelection):

    def __init__(self, epsilon, epsilon_decay, epsilon_end):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end

    def select(self, actions, legal_actions):
        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            return np.argmax(actions)
        else:
            return random.choice(np.indices(actions)[legal_actions])
        self.decay()

    def decay(self):
        # Epsilon decay
        self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon)


class RandomActionSelection(EpsilonGreedyActionSelection):

    def __init__(self):
        super.__init__(epsilon=1, epsilon_decay=1, epsilon_end=1)


class GreedyActionSelection(EpsilonGreedyActionSelection):

    def __init__(self):
        super.__init__(epsilon=0, epsilon_decay=0, epsilon_end=0)


def BoltzmannActionSelection(CategoricalActionSelection):

    def __init__(self, temperature, temperature_decay, temperature_end):
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.temperature_end = temperature_end

    def decay(self):
        self.temperature = max(
            self.temperature_end,
            self.temperature_decay * self.temperature
        )


class CategoricalActionSelection(ActionSelection):

    def __init__(self):
        super.__init__(self, temperature=1,
                       temperature_decay=1, temperature_end=1)

    def select(actions, legal_actions):
        return random.choice(np.indices(actions)[legal_actions], p=actions[legal_actions])
