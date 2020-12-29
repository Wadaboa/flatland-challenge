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
from model.models import DQN, DuelingDQN, EntireGNN, MultiGNN
from policy.replay_buffers import ReplayBuffer


LOSSES = {
    "huber": policy_utils.MaskedHuberLoss(),
    "mse": policy_utils.MaskedMSELoss(),
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
        super(RandomPolicy, self).__init__(
            params, state_size, choice_size=env_utils.RailEnvChoices.choice_size(),
            choice_selector=RandomActionSelector(), training=training
        )

    def act(self, states, legal_choices, moving_agents, training=False):
        choice_values = np.zeros((moving_agents.shape[0], self.choice_size))
        return self.choice_selector.select_many(
            choice_values, moving_agents, np.array(legal_choices),
            training=(training and self.training)
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
        net = DuelingDQN if self.params.model.dqn.dueling.enabled else DQN
        self.qnetwork_local = net(
            self.state_size, env_utils.RailEnvChoices.choice_size(),
            self.params.model.dqn, device=self.device
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

    def act(self, states, legal_choices, moving_agents, training=False):
        '''
        Perform action selection based on the Q-values returned by the network
        '''
        choice_values = np.zeros((moving_agents.shape[0], self.choice_size),)
        # If no agent is active, then skip computations
        if moving_agents.any():

            # Add 1 dimension to state to simulate a mini-batch of size 1
            if self.params.policy.type.decentralized_fov:
                states = Batch.from_data_list([states[0]]).to(self.device)
            elif self.params.policy.type.graph:
                states = Batch.from_data_list(states).to(self.device)
            else:
                states = torch.tensor(
                    states, dtype=torch.float, device=self.device
                )

            # Convert moving agents to tensor
            t_moving_agents = torch.from_numpy(
                moving_agents
            ).bool().to(self.device)

            # Call the network
            self.qnetwork_local.eval()
            with torch.no_grad():
                choice_values = self.qnetwork_local(
                    states, mask=t_moving_agents
                ).detach().cpu().numpy()

        # Select a legal choice based on the action selector
        return self.choice_selector.select_many(
            choice_values, moving_agents, np.array(legal_choices),
            training=(training and self.training)
        )

    def step(self, experiences):
        '''
        Add an experience to memory and eventually perform a training step
        '''
        assert self.training, "Policy has been initialized for evaluation only"
        for experience in experiences:
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
        states, choices, rewards, next_states, next_legal_choices, finished, moving = experiences

        # Get expected Q-values from local model
        q_expected = self.qnetwork_local(states, mask=moving).gather(
            1, choices.flatten().unsqueeze(1)
        ).squeeze(1)

        # Get expected Q-values from target model
        q_targets_next = torch.from_numpy(
            self._get_q_targets_next(
                next_states, next_legal_choices.cpu().numpy(), moving
            )
        ).squeeze(1).to(self.device)

        # Compute Q-targets for current states
        q_targets = (
            torch.flatten(rewards) + (
                self.params.learning.discount *
                q_targets_next * (1 - torch.flatten(finished))
            )
        )

        # Compute and minimize the loss
        self.loss = self.criterion(q_expected, q_targets, mask=moving)
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

        '''
        print(
            q_expected.shape, 
            q_targets_next.shape, 
            q_targets.shape, 
            rewards.shape, 
            finished.shape, 
            self.loss
        )
        '''

        # Log loss to wandb
        if self.params.generic.enable_wandb and self.params.generic.wandb_gradients.enabled:
            wandb.log({"loss": self.loss})

        # Update target network
        self._soft_update(self.qnetwork_local, self.qnetwork_target)

    def _get_q_targets_next(self, next_states, next_legal_choices, moving):
        '''
        Get expected Q-values from target network
        '''

        def _double_dqn():
            q_targets_next = self.qnetwork_target(
                next_states, mask=moving
            ).detach().cpu().numpy()
            q_locals_next = self.qnetwork_local(
                next_states, mask=moving
            ).detach().cpu().numpy()

            # Softmax Bellman
            if self.params.learning.softmax_bellman.enabled:
                return np.sum(
                    q_targets_next * policy_utils.masked_softmax(
                        q_locals_next,
                        next_legal_choices.reshape(q_locals_next.shape),
                        temperature=self.params.learning.softmax_bellman.temperature
                    ), axis=1, keepdims=True
                )

            # Standard Bellman
            best_choices = policy_utils.masked_argmax(
                q_locals_next,
                next_legal_choices.reshape(q_locals_next.shape)
            )
            return np.take_along_axis(q_targets_next, best_choices, axis=1)

        def _dqn():
            q_targets_next = self.qnetwork_target(
                next_states
            ).detach().cpu().numpy()

            # Standard or softmax Bellman
            return (
                policy_utils.masked_max(
                    q_targets_next,
                    next_legal_choices.reshape(q_targets_next.shape)
                )
                if not self.params.learning.softmax_bellman.enabled
                else np.sum(
                    q_targets_next * policy_utils.masked_softmax(
                        q_targets_next,
                        next_legal_choices.reshape(q_targets_next.shape),
                        temperature=self.params.learning.softmax_bellman.temperature
                    ), axis=1, keepdims=True
                )
            )

        return _double_dqn() if self.params.model.dqn.double else _dqn()

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
                torch.load(filename + ".local", map_location=self.device)
            )
            if self.training and os.path.exists(filename + ".target"):
                self.qnetwork_target.load_state_dict(
                    torch.load(filename + ".target", map_location=self.device)
                )
        else:
            print("Model not found. Please, check the given path.")

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


class DQNGNNPolicy(DQNPolicy):
    '''
    DQN + GNN policy
    '''

    def __init__(self, params, state_size, choice_selector, training=False):
        '''
        Initialize DQNGNNPolicy object
        '''
        super(DQNGNNPolicy, self).__init__(
            params, (
                params.model.entire_gnn.embedding_size *
                params.model.entire_gnn.pos_size
            ),
            choice_selector, training=training
        )

        self.qnetwork_local = policy_utils.Sequential(
            EntireGNN(
                state_size, self.params.observator.max_depth,
                self.params.model.entire_gnn, device=self.device
            ).to(self.device),
            self.qnetwork_local
        )

        if training:
            self.qnetwork_target = copy.deepcopy(self.qnetwork_local)


class DecentralizedFOVDQNPolicy(DQNPolicy):
    '''
    Decentralized FOV CNN + GNN + DQN policy
    '''

    def __init__(self, params, state_size, choice_selector, training=False):
        '''
        Initialize MultiAgentDQNGNNPolicy object
        '''
        super(DecentralizedFOVDQNPolicy, self).__init__(
            params, params.model.multi_gnn.gnn_communication.embedding_size,
            choice_selector, training=training
        )

        self.qnetwork_local = policy_utils.Sequential(
            MultiGNN(
                self.params.observator.max_depth,
                self.params.observator.max_depth,
                state_size, self.params.model.multi_gnn,
                device=self.device
            ).to(self.device),
            self.qnetwork_local
        )

        if training:
            self.qnetwork_target = copy.deepcopy(self.qnetwork_local)


POLICIES = {
    "tree": DQNPolicy,
    "binary_tree": DQNPolicy,
    "graph": DQNGNNPolicy,
    "decentralized_fov": DecentralizedFOVDQNPolicy,
    "random": RandomPolicy
}
