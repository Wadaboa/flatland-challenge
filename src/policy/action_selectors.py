import random

import numpy as np

from policy import policy_utils


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

    def __init__(self, parameter_start, *args):
        super(NullParameterDecay, self).__init__(
            parameter_start, parameter_start, parameter_decay=0
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


PARAMETER_DECAYS = {
    "none": NullParameterDecay,
    "linear": LinearParameterDecay,
    "exponential": ExponentialParameterDecay
}


class ActionSelector:

    def __init__(self, decay_schedule):
        assert isinstance(decay_schedule, ParameterDecay)
        self.decay_schedule = decay_schedule

    def select(self, actions, legal_actions=None, training=False):
        raise NotImplementedError()

    def select_many(self, actions, moving_agents, legal_actions, training=False):
        assert len(moving_agents.shape) == 1
        assert len(legal_actions.shape) == 2
        assert len(actions.shape) == 2
        assert moving_agents.shape[0] == legal_actions.shape[0] == actions.shape[0]
        assert legal_actions.shape[1] == actions.shape[1]
        num_agents = moving_agents.shape[0]
        choices = np.full((num_agents,), -1)
        is_best = np.full((num_agents,), False)
        for handle in range(num_agents):
            if moving_agents[handle]:
                choices[handle], is_best[handle] = self.select(
                    actions[handle], legal_actions[handle], training=training
                )
        return choices, is_best

    def decay(self):
        return None

    def reset(self):
        return None

    def get_parameter(self):
        return None


class EpsilonGreedyActionSelector(ActionSelector):

    def __init__(self, decay_schedule):
        super(EpsilonGreedyActionSelector, self).__init__(decay_schedule)
        self.epsilon = decay_schedule.parameter_start

    def select(self, actions, legal_actions=None, training=False):
        if legal_actions is None:
            legal_actions = np.ones_like(actions, dtype=bool)
        max_action = policy_utils.masked_argmax(actions, legal_actions, dim=0)
        if not training or random.random() > self.epsilon:
            return max_action, True
        random_action = np.random.choice(
            np.arange(actions.size)[legal_actions]
        )
        return (random_action, max_action == random_action)

    def decay(self):
        self.epsilon = self.decay_schedule.decay(self.epsilon)

    def reset(self):
        self.epsilon = self.decay_schedule.parameter_start

    def get_parameter(self):
        return self.epsilon


class RandomActionSelector(EpsilonGreedyActionSelector):

    def __init__(self, *args):
        super(RandomActionSelector, self).__init__(
            NullParameterDecay(parameter_start=1)
        )


class GreedyActionSelector(EpsilonGreedyActionSelector):

    def __init__(self, *args):
        super(GreedyActionSelector, self).__init__(
            NullParameterDecay(parameter_start=0)
        )


class BoltzmannActionSelector(ActionSelector):

    def __init__(self, decay_schedule):
        super(BoltzmannActionSelector, self).__init__(decay_schedule)
        self.temperature = decay_schedule.parameter_start

    def select(self, actions, legal_actions=None, training=False):
        if legal_actions is None:
            legal_actions = np.ones_like(actions, dtype=bool)
        max_action = policy_utils.masked_argmax(actions, legal_actions, dim=0)
        if not training:
            return max_action, True
        dist = policy_utils.masked_softmax(
            actions, legal_actions, dim=0, temperature=self.temperature
        )
        random_action = np.random.choice(
            np.arange(actions.size)[legal_actions], p=dist[legal_actions]
        )
        is_equal = max_action == random_action
        return random_action, is_equal

    def decay(self):
        self.temperature = self.decay_schedule.decay(self.temperature)

    def reset(self):
        self.temperature = self.decay_schedule.parameter_start

    def get_parameter(self):
        return self.temperature


class CategoricalActionSelector(BoltzmannActionSelector):

    def __init__(self, *args):
        super(CategoricalActionSelector, self).__init__(
            NullParameterDecay(parameter_start=1)
        )


ACTION_SELECTORS = {
    "eps_greedy":  EpsilonGreedyActionSelector,
    "random": RandomActionSelector,
    "greedy": GreedyActionSelector,
    "boltzmann": BoltzmannActionSelector,
    "categorical": CategoricalActionSelector
}
