import random

import numpy as np

import model_utils


class ActionSelector:

    def select(actions, legal_actions=None):
        pass


class EpsilonGreedyActionSelector(ActionSelector):

    def __init__(self, epsilon, epsilon_decay, epsilon_end):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end

    def select(self, actions, legal_actions=None):
        legal_actions = (
            np.ones_like(actions, dtype=bool) if legal_actions is None
            else legal_actions
        )
        if random.random() > self.epsilon:
            return np.argmax(actions)
        else:
            return random.choice(np.indices(actions)[legal_actions])

    def decay(self):
        self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon)


class RandomActionSelector(EpsilonGreedyActionSelector):

    def __init__(self):
        super.__init__(epsilon=1, epsilon_decay=1, epsilon_end=1)


class GreedyActionSelector(EpsilonGreedyActionSelector):

    def __init__(self):
        super.__init__(epsilon=0, epsilon_decay=0, epsilon_end=0)


def BoltzmannActionSelector(ActionSelector):

    def __init__(self, temperature, temperature_decay, temperature_end):

        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.temperature_end = temperature_end

    def select(actions, legal_actions=None):
        legal_actions = (
            np.ones_like(actions, dtype=bool) if legal_actions is None
            else legal_actions
        )
        dist = model_utils.masked_softmax(
            actions, legal_actions, dim=0, temperature=self.temperature
        )
        return random.choice(np.indices(actions)[legal_actions], p=dist[legal_actions])

    def decay(self):
        self.temperature = max(
            self.temperature_end,
            self.temperature_decay * self.temperature
        )


class CategoricalActionSelector(BoltzmannActionSelector):

    def __init__(self):
        super(CategoricalActionSelector, self).__init__(
            temperature=1, temperature_decay=1, temperature_end=1
        )
