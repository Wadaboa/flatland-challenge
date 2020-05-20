import numpy as np

from flatland.envs.rail_env import RailEnvActions


class RandomAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

    def act(self, state):
        return np.random.choice([
            RailEnvActions.MOVE_FORWARD,
            RailEnvActions.MOVE_LEFT,
            RailEnvActions.MOVE_RIGHT
        ])

    def step(self, memories):
        return

    def save(self, filename):
        return

    def load(self, filename):
        return


class SimpleAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

    def act(self, state):
        print(state)
        _, predictions, _ = state
        if predictions is None:
            return RailEnvActions.DO_NOTHING

        _, edges, _ = predictions
        _, _, data = edges[0]
        print(data['action'])
        return data['action']

    def step(self, memories):
        return

    def save(self, filename):
        return

    def load(self, filename):
        return
