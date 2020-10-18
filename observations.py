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
- Shortest path action (one-hot encoding)
- Shortest path distance
- Deviation path action (one-hot encoding)
- Deviation path distance
- Neighboring agents in the same direction (in the packed graph)
- Distance from neighboring agents in the same direction (in the packed graph)
- Neighboring agents in the opposite direction (in the packed graph)
- Distance from neighboring agents in the opposite direction (in the packed graph)
- Number of conflicts in the shortest path
'''

# TODO
'''
    -Graph:
        - Divide intersection nodes on arrival direction
        -
    - Observation:
        - Structure:
            - Shortest path cut on the 5th node
            - Foreach node on Shortest path alternative route lenght max 5
        - Features node:
            - Num. agents on the previous path in the same direction
            - Min distance agent on the previous path in the same direction from agent root
            - Num. agents on the previous path in the opposite direction
            - Min distance agent on the previous path in the opposite direction from agent root
            - Distance from the target
            - Num. agents using the node to reach their target in the shortest path
            - Distance from the root agent and a possible conflict with another agent in the previous path
            - Deadlock probabily in the previous path
            - Number of agent malfunctioning in the previous path
            - Turns to wait blocked if there is an agent on path malfunctioning (difference between malfunction time and distance)
'''


class CustomObservation(ObservationBuilder):

    FEATURES = 10

    def __init__(self, max_depth, predictor):
        super().__init__()
        self.max_depth = max_depth
        self.predictor = predictor
        self.collisions = dict()
        self.observations = dict()

    def reset(self):
        self.railway_encoding = CellOrientationGraph(
            grid=self.env.rail.grid, agents=self.env.agents
        )
        # print(self.railway_encoding.graph.edges.data())
        # self.railway_encoding.draw_graph()
        if self.predictor:
            self.predictor.set_railway_encoding(self.railway_encoding)
            self.predictor.reset()

    def get_many(self, handles=None):
        self.predictions = self.predictor.get_many()
        self.find_collisions()

        # add malfunctions info

        return super().get_many(handles)

    def get(self, handle=0):
        '''
        # Consider agent speed
        position = self.railway_encoding.get_agent_cell(handle)
        agent_speed = agent.speed_data["speed"]
        times_per_cell = int(np.reciprocal(agent_speed))
        remaining_steps = int(
            (1 - agent.speed_data["position_fraction"]) / agent_speed
        )

        # Edit weights to account for agent speed
        for edge in edges:
            edge[2]['distance'] = edge[2]['weight'] * times_per_cell
        edges[0][2]['distance'] -= (times_per_cell - remaining_steps)

        # Edit positions to account for agent speed
        positions = [pos[0]] * (remaining_steps)
        for position in pos[1:]:
            positions.extend([position] * times_per_cell)
        '''

        self.observations[handle] = np.ones(
            (self.max_depth, self.max_depth, self.FEATURES)
        ) * -np.inf
        if self.predictions[handle] is not None:
            shortest_path_prediction, deviation_paths_prediction = self.predictions[handle]

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
