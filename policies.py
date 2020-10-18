import numpy as np

from flatland.envs.rail_env import RailEnvActions


class Policy:

    def __init__(self, state_size=None, action_size=None):
        self.state_size = state_size
        self.action_size = action_size

    def act(self, state):
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
