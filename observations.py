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
        self.collisions = dict()
        self.observations = dict()

    def reset(self):
        self.railway_encoding = CellOrientationGraph(
            grid=self.env.rail.grid, agents=self.env.agents
        )
        print(self.railway_encoding.graph.edges.data())
        if self.predictor:
            self.predictor.set_railway_encoding(self.railway_encoding)
            self.predictor.reset()

    def get_many(self, handles=None):
        self.predictions = self.predictor.get()
        self.find_collisions()

        # add malfunctions info

        return super().get_many(handles)

    def get(self, handle=0):
        if handle not in self.observations or not self.observations[handle]:
            '''
            self.observations[handle] = self.railway_encoding.meaningful_subgraph(
                handle
            )
            '''
            pass

        self.observations[handle] = (
            self.railway_encoding,
            self.predictions[handle],
            self.agent_collisions(handle)
        )

        self.env.dev_obs_dict[handle] = OrderedSet()

        return self.observations[handle]

    def find_collisions(self):
        '''
        Check for future crashes and deadlocks
        '''
        positions = []
        for pred in self.predictions.values():
            if pred is not None:
                _, _, pos = pred
                positions.append(list(pos))
            else:
                positions.append([None] * self.predictor.max_depth)
        positions = utils.fill_none(positions, lenght=self.predictor.max_depth)
        for t, col in enumerate(map(list, zip(*positions))):
            dups = utils.find_duplicates(col)
            if dups:
                self.collisions[t] = dups

        self.find_deadlocks(positions)

    def find_deadlocks(self, positions):
        '''
        Check for future deadlocks
        '''
        pos = list(map(list, zip(*positions)))
        for t, col in enumerate(pos):
            if t - 1 >= 0:
                old_col = pos[t - 1]
                for i, elem in enumerate(col):
                    dups = utils.find_duplicate(old_col, elem, i)
                    if dups:
                        self.collisions[t] = dups

    def agent_collisions(self, handle):
        '''
        Re-index collisions based on agent handles
        '''
        meaningful_collisions = dict()
        for t, dups in self.collisions.items():
            for pos, agents in dups:
                if handle in agents:
                    meaningful_collisions[pos] = (
                        t, agents.difference({handle})
                    )
        return meaningful_collisions

    def set_env(self, env):
        super().set_env(env)
        if self.predictor:
            self.predictor.set_env(self.env)
