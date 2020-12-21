import os
import copy

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import wandb
from torch_geometric.data import Data, Batch

from env import env_utils
from policy import policy_utils
from policy.action_selectors import ActionSelector, RandomActionSelector
from model.models import DQN, DuelingDQN, SingleDQNGNN, MultiDQNGNN
from policy.replay_buffers import ReplayBuffer


LOSSES = {
    "huber": nn.SmoothL1Loss(),
    "mse": nn.MSELoss(),
    "masked_mse": policy_utils.MaskedMSELoss()
}


class Policy:
    '''
    Policy abstract class
    '''

    def __init__(self, params=None, state_size=None, choice_size=None, choice_selector=None, training=False):
        self.params = params
        self.state_size = state_size
        self.choice_size = choice_size
        self.choice_selector = choice_selector
        self.training = training

    def act(self, state, legal_choices=None, training=False):
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

    def __init__(self, params=None, state_size=None, choice_selector=None, training=False):
        self.action_selector = RandomActionSelector()
        super(RandomPolicy, self).__init__(
            params, state_size, choice_size=env_utils.RailEnvChoices.choice_size(),
            choice_selector=choice_selector, training=training
        )

    def act(self, state, legal_choices=None, training=False):
        choices = np.zeros((len(env_utils.RailEnvChoices.values())))
        legal_choices = np.full(
            choices.shape, legal_choices, dtype=bool
        )
        return self.action_selector.select(
            choices, legal_choices, training=training
        )

    def step(self, experience):
        return None

    def save(self, filename):
        return None

    def load(self, filename):
        return None


class DQNPolicy(Policy):
    '''
    DQN policy
    '''

    def __init__(self, params, state_size, choice_selector, training=False):
        '''
        Initialize DQNPolicy object
        '''
        super(DQNPolicy, self).__init__(
            params, state_size, choice_size=env_utils.RailEnvChoices.choice_size(),
            choice_selector=choice_selector, training=training
        )
        assert isinstance(
            choice_selector, ActionSelector
        ), "The choice selection object must be an instance of ActionSelector"

        # Parameters
        self.device = torch.device("cpu")
        if self.params.generic.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("🐇 Using GPU")

        # Q-Network
        net = DuelingDQN if self.params.model.dueling else DQN
        self.qnetwork_local = net(
            self.state_size, env_utils.RailEnvChoices.choice_size(),
            hidden_sizes=self.params.model.hidden_sizes,
            nonlinearity=self.params.model.nonlinearity.get_true_key()
        ).to(self.device)

        # Training parameters
        if self.training:
            self.time_step = 0
            self.qnetwork_target = copy.deepcopy(self.qnetwork_local)
            self.optimizer = optim.Adam(
                self.qnetwork_local.parameters(), lr=self.params.learning.learning_rate
            )
            self.criterion = LOSSES[self.params.learning.loss.get_true_key()]
            self.loss = torch.tensor(0.0)
            self.memory = ReplayBuffer(
                env_utils.RailEnvChoices.choice_size(), self.params.replay_buffer.batch_size,
                self.params.replay_buffer.size, self.device
            )

    def enable_wandb(self):
        '''
        Log gradients and parameters to wandb
        '''
        wandb.watch(
            self.qnetwork_local, self.criterion,
            log="all", log_freq=self.params.generic.wandb_gradients.checkpoint
        )

    def act(self, state, legal_choices, training=False):
        '''
        Perform action selection based on the Q-values returned by the network
        '''
        # Add 1 dimension to state to simulate a mini-batch of size 1
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(
                state
            ).float().unsqueeze(0).to(self.device)
        elif isinstance(state, Data):
            state = Batch.from_data_list([state]).to(self.device)

        # Call the network
        self.qnetwork_local.eval()
        with torch.no_grad():
            choice_values = self.qnetwork_local(
                state
            ).squeeze().detach().cpu().numpy()

        # Select a legal choice based on the action selector
        legal_choices = np.array(legal_choices)
        return self.choice_selector.select(
            choice_values, legal_choices, training=(training and self.training)
        )

    def step(self, experience):
        '''
        Add an experience to memory and eventually perform a training step
        '''
        assert self.training, "Policy has been initialized for evaluation only"

        # Save experience in replay memory
        self.memory.add(experience)

        # Learn every `checkpoint` time steps
        # (if enough samples are available in memory, get random subset and learn)
        self.time_step = (
            self.time_step + 1
        ) % self.params.replay_buffer.checkpoint
        if self.time_step == 0 and self.memory.can_sample():
            self.qnetwork_local.train()
            self._learn()

    def _learn(self):
        '''
        Perform a learning step
        '''
        # Sample a batch of experiences
        experiences = self.memory.sample()
        states, choices, rewards, next_states, next_legal_choices, finished = experiences

        # Get expected Q-values from local model
        q_expected = self.qnetwork_local(states).gather(1, choices)

        # Get expected Q-values from target model
        q_targets_next = torch.from_numpy(
            self._get_q_targets_next(
                next_states, next_legal_choices.cpu().numpy()
            )
        ).to(self.device)

        # Compute Q-targets for current states
        q_targets = (
            rewards + (
                self.params.learning.discount *
                q_targets_next * (1 - finished)
            )
        )

        # Compute and minimize the loss
        self.loss = self.criterion(q_expected, q_targets)
        self.optimizer.zero_grad()
        self.loss.backward()
        if self.params.learning.gradient.clip_norm:
            nn.utils.clip_grad.clip_grad_norm_(
                self.qnetwork_local.parameters(), self.params.learning.gradient.max_norm
            )
        elif self.params.learning.gradient.clamp_values:
            nn.utils.clip_grad.clip_grad_value_(
                self.qnetwork_local.parameters(), self.params.learning.gradient.value_limit
            )
        self.optimizer.step()

        # Log loss to wandb
        if self.params.generic.enable_wandb and self.params.generic.wandb_gradients.enabled:
            wandb.log({"loss": self.loss})

        # Update target network
        self._soft_update(self.qnetwork_local, self.qnetwork_target)

    def _get_q_targets_next(self, next_states, next_legal_choices):
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
            if self.params.learning.softmax_bellman:
                return np.sum(
                    q_targets_next * policy_utils.masked_softmax(
                        q_locals_next, next_legal_choices
                    ), axis=1, keepdims=True
                )

            # Standard Bellman
            best_choices = policy_utils.masked_argmax(
                q_targets_next, next_legal_choices
            )
            return np.take_along_axis(q_targets_next, best_choices, axis=1)

        def _dqn():
            q_targets_next = self.qnetwork_target(
                next_states
            ).detach().cpu().numpy()

            # Standard or softmax Bellman
            return (
                policy_utils.masked_max(q_targets_next, next_legal_choices)
                if not self.params.learning.softmax_bellman
                else np.sum(
                    policy_utils.masked_softmax(
                        q_targets_next, next_legal_choices
                    ) * q_targets_next, axis=1, keepdims=True
                )
            )

        return _double_dqn() if self.params.model.double else _dqn()

    def _soft_update(self, local_model, target_model):
        '''
        Soft update model parameters: θ_target = τ * θ_local + (1 - τ) * θ_target
        '''
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                self.params.learning.tau * local_param.data +
                (1.0 - self.params.learning.tau) * target_param.data
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
        else:
            print("Model not found - check given path")

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


class SingleAgentDQNGNNPolicy(DQNPolicy):
    '''
    Single agent DQN + GNN policy
    '''

    def __init__(self, params, state_size, choice_selector, training=False):
        '''
        Initialize SingleAgentDQNGNNPolicy object
        '''
        super(SingleAgentDQNGNNPolicy, self).__init__(
            params, state_size, choice_selector, training=training
        )

        # Q-Network
        self.qnetwork_local = SingleDQNGNN(
            state_size, env_utils.RailEnvChoices.choice_size(),
            self.params.model.single_gnn.pos_size,
            self.params.model.single_gnn.embedding_size,
            hidden_sizes=self.params.model.hidden_sizes,
            nonlinearity=self.params.model.nonlinearity.get_true_key(),
            gnn_hidden_size=self.params.model.single_gnn.hidden_size,
            depth=self.params.observator.max_depth,
            dropout=self.params.model.single_gnn.dropout
        ).to(self.device)

        if training:
            self.qnetwork_target = copy.deepcopy(self.qnetwork_local)


class MultiAgentDQNGNNPolicy(DQNPolicy):
    '''
    Multi agent DQN + GNN policy
    '''

    def __init__(self, params, state_size, choice_selector, training=False):
        '''
        Initialize MultiAgentDQNGNNPolicy object
        '''
        super(MultiAgentDQNGNNPolicy, self).__init__(
            params, params.model.multi_gnn.embedding_size,
            choice_selector, training=training
        )

        # Q-Network
        self.qnetwork_local = MultiDQNGNN(
            env_utils.RailEnvChoices.choice_size(),
            self.params.observator.max_depth,
            self.params.observator.max_depth,
            state_size,
            self.params.model.multi_gnn.output_channels,
            hidden_channels=self.params.model.multi_gnn.hidden_channels,
            pool=self.params.model.multi_gnn.pool,
            embedding_size=self.params.model.multi_gnn.embedding_size,
            hidden_sizes=self.params.model.hidden_sizes,
            nonlinearity=self.params.model.nonlinearity.get_true_key(),
            device=self.device
        ).to(self.device)

        if training:
            self.qnetwork_target = copy.deepcopy(self.qnetwork_local)

    def act(self, state, legal_choices, adjacency, inactives, training=False):
        '''
        Perform action selection based on the Q-values returned by the network
        '''
        # Add 1 dimension to state to simulate a mini-batch of size 1
        state = torch.from_numpy(
            state
        ).float().unsqueeze(0).to(self.device)
        adjacency = torch.from_numpy(
            adjacency
        ).int().unsqueeze(0).to(self.device)
        inactives = torch.from_numpy(
            inactives
        ).bool().unsqueeze(0).to(self.device)

        # Call the network
        self.qnetwork_local.eval()
        with torch.no_grad():
            choice_values = self.qnetwork_local(
                state, adjacency, inactives
            ).squeeze().detach().cpu().numpy()

        # Select a legal choice based on the action selector
        num_agents = adjacency.shape[1]
        actions = np.full((num_agents,), -1)
        is_best = np.full((num_agents,), False)
        for handle in range(num_agents):
            if not inactives[0, handle]:
                actions[handle], is_best[handle] = self.choice_selector.select(
                    choice_values[handle], np.array(legal_choices[handle]),
                    training=(training and self.training)
                )

        return actions, is_best

    def step(self, experience):
        '''
        Add an experience to memory and eventually perform a training step
        '''
        assert self.training, "Policy has been initialized for evaluation only"

        # Save experience in replay memory
        self.memory.add(experience, multi=True)

        # Learn every `checkpoint` time steps
        # (if enough samples are available in memory, get random subset and learn)
        self.time_step = (
            self.time_step + 1
        ) % self.params.replay_buffer.checkpoint
        if self.time_step == 0 and self.memory.can_sample():
            self.qnetwork_local.train()
            self._learn()

    def _learn(self):
        '''
        Perform a learning step
        '''
        # Sample a batch of experiences
        experiences = self.memory.sample(multi=True)
        states, choices, adjacencies, rewards, next_states, next_legal_choices, next_adjacencies, finished, inactives = experiences

        # Get expected Q-values from local model
        # (batch_size, num_agents, choice_size)
        q_expected = self.qnetwork_local(
            states, adjacencies, inactives
        )
        # Gather to (batch_size, num_agents, 1)
        q_expected = q_expected.gather(2, choices.unsqueeze(2)).squeeze(2)

        # Get expected Q-values from target model
        q_targets_next = torch.from_numpy(
            self._get_q_targets_next(
                next_states, next_legal_choices.cpu().numpy(),
                next_adjacencies, inactives
            )
        ).squeeze(2).to(self.device)

        # Compute Q-targets for current states
        q_targets = (
            rewards + (
                self.params.learning.discount *
                q_targets_next * (1 - finished)
            )
        )

        # Compute and minimize the loss
        self.loss = self.criterion(q_expected, q_targets, inactives)
        self.optimizer.zero_grad()
        self.loss.backward()
        if self.params.learning.gradient.clip_norm:
            nn.utils.clip_grad.clip_grad_norm_(
                self.qnetwork_local.parameters(), self.params.learning.gradient.max_norm
            )
        elif self.params.learning.gradient.clamp_values:
            nn.utils.clip_grad.clip_grad_value_(
                self.qnetwork_local.parameters(), self.params.learning.gradient.value_limit
            )
        self.optimizer.step()

        # Log loss to wandb
        if self.params.generic.enable_wandb and self.params.generic.wandb_gradients.enabled:
            wandb.log({"loss": self.loss})

        # Update target network
        self._soft_update(self.qnetwork_local, self.qnetwork_target)

    def _get_q_targets_next(self, next_states, next_legal_choices, next_adjacencies, inactives):
        '''
        Get expected Q-values from target network
        '''

        def _double_dqn():
            q_locals_next = self.qnetwork_local(
                next_states, next_adjacencies, inactives
            ).detach().cpu().numpy()
            q_targets_next = self.qnetwork_target(
                next_states, next_adjacencies, inactives
            ).detach().cpu().numpy()

            # Softmax Bellman
            if self.params.learning.softmax_bellman:
                return np.sum(
                    q_targets_next * policy_utils.masked_softmax(
                        q_locals_next, next_legal_choices, dim=2
                    ), axis=2, keepdims=True
                )

            # Standard Bellman
            best_choices = policy_utils.masked_argmax(
                q_targets_next, next_legal_choices, dim=2
            )
            return np.take_along_axis(q_targets_next, best_choices, axis=2)

        def _dqn():
            q_targets_next = self.qnetwork_target(
                next_states
            ).detach().cpu().numpy()

            # Standard or softmax Bellman
            return (
                policy_utils.masked_max(q_targets_next, next_legal_choices)
                if not self.params.learning.softmax_bellman
                else np.sum(
                    q_targets_next * policy_utils.masked_softmax(
                        q_targets_next, next_legal_choices, dim=2
                    ), axis=2, keepdims=True
                )
            )

        return _double_dqn() if self.params.model.double else _dqn()


POLICIES = {
    "tree": DQNPolicy,
    "binary_tree": DQNPolicy,
    "single_agent_graph": SingleAgentDQNGNNPolicy,
    "multi_agent_graph": MultiAgentDQNGNNPolicy,
    "random": RandomPolicy
}
