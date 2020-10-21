'''
'''

from collections import namedtuple

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
            - [Y] Num. agents on the previous path in the same direction
            - [Y] Min distance agent on the previous path in the same direction from agent root
            - [Y] Num. agents on the previous path in the opposite direction
            - [Y] Min distance agent on the previous path in the opposite direction from agent root
            - [Y] Distance from the target
            - [Y] Num. agents using the node to reach their target in the shortest path
            - Distance from the root agent and a possible conflict with another agent in the previous path
            - Deadlock probabily in the previous path
            - [Y] Number of agent malfunctioning in the previous path in the same direction
            - [Y] Number of agent malfunctioning in the previous path in the opposite direction
            - [Y] Turns to wait blocked if there is an agent on path malfunctioning in the same direction (difference between malfunction time and distance)
            - [Y] Turns to wait blocked if there is an agent on path malfunctioning in the opposite direction (difference between malfunction time and distance)
'''

SpeedData = namedtuple('SpeedData', ['times', 'remaining'])


class CustomObservation(ObservationBuilder):

    FEATURES = 10

    def __init__(self, max_depth, predictor):
        super().__init__()
        self.max_depth = max_depth
        self.predictor = predictor
        self.observations = dict()
        self.agent_handles = set()
        self.other_agents = dict()
        self.speed_data = dict()

    def _init_env(self):
        self.railway_encoding = CellOrientationGraph(
            grid=self.env.rail.grid, agents=self.env.agents
        )
        if self.predictor:
            self.predictor.set_railway_encoding(self.railway_encoding)
            self.predictor.reset()

    def _init_speed_data(self):
        for handle, agent in enumerate(self.env.agents):
            times_per_cell = int(np.reciprocal(agent.speed_data["speed"]))
            self.speed_data[handle] = SpeedData(
                times=times_per_cell, remaining=0
            )

    def _init_agents(self):
        self.agent_handles = set(self.env.get_agent_handles())
        self.other_agents = {
            h: self.agent_handles - {h}
            for h in self.agent_handles
        }

    def reset(self):
        self._init_env()
        self._init_agents()
        self._init_speed_data()

    def get_many(self, handles=None):
        self.predictions = self.predictor.get_many()
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

            remaining_steps = int(
                (1 - self.env.agents[handle].speed_data["position_fraction"])
                / self.env.agents[handle].speed_data["speed"]
            )
            self.speed_data[handle] = SpeedData(
                times=self.speed_data[handle].times,
                remaining=remaining_steps
            )

            shortest_path_prediction, deviation_paths_prediction = self.predictions[handle]
            cum_weights = self.compute_cumulative_weights(
                shortest_path_prediction.edges, remaining_steps
            )
            num_agents, distances = self.agents_in_path(
                handle,
                shortest_path_prediction.path,
                cum_weights
            )
            target_distances = self.distance_from_target(
                handle,
                shortest_path_prediction.lenght,
                shortest_path_prediction.path,
                cum_weights
            )
            other_shortest_paths = []
            for agent in self.other_agents[handle]:
                s, _ = self.predictions[agent]
                other_shortest_paths.append(s.path)
            c_nodes = self.common_nodes(
                shortest_path_prediction.path, other_shortest_paths
            )

        return self.observations[handle]

    def compute_cumulative_weights(self, handle, edges, initial_distance):
        '''
        Given a list of edges, compute the cumulative sum of weights
        '''
        weights = [initial_distance] + [
            e[2]['weight'] * self.speed_data[handle].times for e in edges
        ]
        return np.cumsum(weights)

    def agents_in_path(self, handle, path, cum_weights):
        '''
        Return two arrays, with the same lenght as the given path, s.t.
        the first array contains the number of agents identified in the subpath
        from the root up to each node in the path (in both directions);
        while the second array contains the distance from the root
        to the closest agent in the subpath (in both directions),
        still up to each node in the path
        '''
        num_agents = np.zeros((len(path), 4))
        distances = np.ones((len(path), 4)) * np.inf
        for agent in self.other_agents[handle]:
            directions = [0]
            position = self.railway_encoding.get_agent_cell(agent)
            node, remaining_distance = self.railway_encoding.next_node(
                position
            )
            nodes = self.railway_encoding.get_nodes((node[0], node[1]))
            for other_node in nodes:
                index = utils.get_index(path, other_node)
                if index:
                    if other_node != node:
                        directions = [1]
                    malfunction = self.env.agents[agent].malfunction_data['malfunction']
                    if malfunction > 0:
                        directions.append(directions[0] + 2)
                    break
            if index:
                for direction in directions:
                    num_agents[index:][direction] += 1
                    value = (cum_weights[index] +
                             self.speed_data[agent].remaining)
                    if direction % 2 == 0:
                        value -= (
                            remaining_distance * self.speed_data[handle].times
                        )
                    else:
                        value += (
                            remaining_distance * self.speed_data[handle].times
                        )
                    if direction >= 2:
                        value = np.clip(malfunction - value, 0, None)
                    if ((direction < 2 and distances[index][direction] > value) or
                            (direction >= 2 and distances[index][direction] < value)):
                        distances[index:][direction] = value
        return num_agents, distances

    def distance_from_target(self, handle, lenght, path, cum_weights):
        '''
        Given the full lenght of a path from a root node to an agent's target,
        compute the distance from each node of the path to the target
        '''
        distances = (
            np.ones((len(path),))
            * lenght
            * self.speed_data[handle].times
        )
        distances = (
            (distances - cum_weights)
            + 2 * self.speed_data[handle].remaining
        )
        return distances

    def common_nodes(self, path, other_paths):
        '''
        Given an agent's path and corresponding paths for every other agent,
        compute the number of intersections for each node
        '''
        nd_other_paths = np.array(other_paths, np.dtype('int, int, int'))
        nd_path = np.array(path, np.dtype('int, int, int'))
        return np.array([np.count_nonzero(nd_other_paths == p) for p in nd_path])

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
