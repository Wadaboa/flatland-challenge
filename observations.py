'''
'''


import numpy as np

from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid_utils import coordinate_to_position
from flatland.utils.ordered_set import OrderedSet

import utils
from railway_encoding import CellOrientationGraph


# Observations
'''
- Agent handle
- Agent status
- Agent speed
- Agent moving
- Shortest path action
- Shortest path distance
- Deviation path action
- Deviation path distance
- Neighboring agents in the same direction (in the packed graph)
- Distance from neighboring agents in the same direction (in the packed graph)
- Neighboring agents in the opposite direction (in the packed graph)
- Distance from neighboring agents in the opposite direction (in the packed graph)
- Number of conflicts in the shortest path
'''


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

        shortest_path_lenght, edges, _ = self.predictions[handle]
        shortest_path_action = edges[0][-1]['action']
        neighbors_same_direction = self.railway_encoding.get_neighboring_agents_same_direction(
            handle
        )
        neighbors_opposite_direction = self.railway_encoding.get_neighboring_agents_opposite_direction(
            handle
        )
        self.observations[handle] = (
            handle,
            self.env.agents[handle].status,
            self.env.agents[handle].speed_data["speed"],
            self.env.agents[handle].moving,
            shortest_path_action,
            shortest_path_lenght,
            # deviation path action
            # deviation path lenght
            neighbors_same_direction,
            neighbors_opposite_direction,
            # distance neighbors same direction
            # distance neighbors opposite direction
            # number of conflicts
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
