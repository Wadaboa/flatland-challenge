import random

import numpy as np

import model_utils


class ParameterDecay:

    def __init__(self, parameter_start, parameter_end,
                 parameter_decay=None, total_episodes=None, decaying_episodes=None):
        parameter_decay_choice = parameter_decay is not None
        episodes_decay_choice = total_episodes is not None and decaying_episodes is not None
        assert parameter_decay_choice or episodes_decay_choice

        self.parameter_start = parameter_start
        self.parameter_end = parameter_end
        self.parameter_decay = parameter_decay

    def decay(self, parameter):
        raise NotImplementedError()


class NullParameterDecay(ParameterDecay):

    def __init__(self, parameter):
        super(NullParameterDecay, self).__init__(
            parameter, parameter, parameter_decay=0
        )

    def decay(self, parameter):
        return parameter


class LinearParameterDecay(ParameterDecay):

    def __init__(self, parameter_start, parameter_end,
                 parameter_decay=None, total_episodes=None, decaying_episodes=None):
        super(LinearParameterDecay, self).__init__(
            parameter_start, parameter_end,
            parameter_decay=parameter_decay,
            total_episodes=total_episodes,
            decaying_episodes=decaying_episodes
        )
        if self.parameter_decay is None:
            self.parameter_decay = (
                (self.parameter_start - self.parameter_end) /
                (total_episodes * decaying_episodes)
            )

    def decay(self, parameter):
        return max(
            self.parameter_end, parameter - self.parameter_decay
        )


class ExponentialParameterDecay(ParameterDecay):

    def __init__(self, parameter_start, parameter_end,
                 parameter_decay=None, total_episodes=None, decaying_episodes=None):
        super(ExponentialParameterDecay, self).__init__(
            parameter_start, parameter_end,
            parameter_decay=parameter_decay,
            total_episodes=total_episodes,
            decaying_episodes=decaying_episodes
        )
        if self.parameter_decay is None:
            self.parameter_decay = (
                (self.parameter_end / self.parameter_start) ^
                (1 / (total_episodes * decaying_episodes))
            )

    def decay(self, parameter):
        return max(
            self.parameter_end, parameter * self.parameter_decay
        )


class ActionSelector:

    def __init__(self, decay_schedule):
        assert isinstance(decay_schedule, ParameterDecay)
        self.decay_schedule = decay_schedule

    def select(self, actions, legal_actions=None, val=False):
        raise NotImplementedError()

    def decay(self):
        return None

    def reset(self):
        return None


class EpsilonGreedyActionSelector(ActionSelector):

    def __init__(self, decay_schedule):
        super(EpsilonGreedyActionSelector, self).__init__(self, decay_schedule)
        self.epsilon = decay_schedule.parameter_start

    def select(self, actions, legal_actions=None, val=False):
        legal_actions = (
            np.ones_like(actions, dtype=bool) if legal_actions is None
            else legal_actions
        )
        max_action = model_utils.masked_argmax(actions, legal_actions, dim=0)
        if val:
            return max_action, False
        random_action = random.choice(np.arange(actions.size)[legal_actions])
        is_equal = max_action == random_action
        return (
            max_action, is_equal if random.random() > self.epsilon
            else random_action, is_equal
        )

    def decay(self):
        self.epsilon = self.decay_schedule.decay(self.epsilon)

    def reset(self):
        self.epsilon = self.decay_schedule.parameter_start


class RandomActionSelector(EpsilonGreedyActionSelector):

    def __init__(self):
        super(RandomActionSelector, self).__init__(
            NullParameterDecay(parameter=1)
        )


class GreedyActionSelector(EpsilonGreedyActionSelector):

    def __init__(self):
        super(GreedyActionSelector, self).__init__(
            NullParameterDecay(parameter=0)
        )


class BoltzmannActionSelector(ActionSelector):

    def __init__(self, decay_schedule):
        super(BoltzmannActionSelector, self).__init__(self, decay_schedule)
        self.temperature = decay_schedule.parameter_start

    def select(self, actions, legal_actions=None, val=False):
        legal_actions = (
            np.ones_like(actions, dtype=bool) if legal_actions is None
            else legal_actions
        )
        max_action = model_utils.masked_argmax(actions, legal_actions, dim=0)
        if val:
            return max_action, False
        dist = model_utils.masked_softmax(
            actions, legal_actions, dim=0, temperature=self.temperature
        )
        random_action = random.choice(
            np.arange(actions.size)[legal_actions], p=dist[legal_actions]
        )
        is_equal = max_action == random_action
        return random_action, is_equal

    def decay(self):
        self.temperature = self.decay_schedule.decay(self.temperature)

    def reset(self):
        self.temperature = self.decay_schedule.parameter_start


class CategoricalActionSelector(BoltzmannActionSelector):

    def __init__(self):
        super(CategoricalActionSelector, self).__init__(
            NullParameterDecay(parameter=1)
        )
