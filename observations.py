'''
'''


import numpy as np

from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid_utils import coordinate_to_position
from flatland.utils.ordered_set import OrderedSet

import utils
from railway_encoding import CellOrientationGraph


class CustomObservation(ObservationBuilder):

    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor
        self.collisions = []
        self.observations = dict()

    def reset(self):
        self.railway_encoding = CellOrientationGraph(
            grid=self.env.rail.grid, agents=self.env.agents
        )
        if self.predictor:
            self.predictor.set_railway_encoding(self.railway_encoding)
            self.predictor.reset()

    def get_many(self, handles=None):
        self.predictions = self.predictor.get()
        self.collisions = self.find_collisions()

        # add malfunctions info

        return super().get_many(handles)

    def get(self, handle=0):
        if handle not in self.observations or not self.observations[handle]:
            self.observations[handle] = self.railway_encoding.meaningful_subgraph(
                handle
            )

        #self.env.dev_obs_dict[handle] = visited

        return self.observations[handle]

    def find_collisions(self):
        positions = [pos for _, _, pos in self.predictions.values()]
        positions = utils.fill_none(positions, lenght=self.predictor.max_depth)
        collisions = []
        for t, col in enumerate(np.array(positions).T):
            dups = utils.duplicates(col)
            if dups:
                collisions.append((t, dups))
        return collisions

    def set_env(self, env):
        super().set_env(env)
        if self.predictor:
            self.predictor.set_env(self.env)
