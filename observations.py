'''
'''

from collections import namedtuple

import numpy as np

from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid_utils import coordinate_to_position
from flatland.utils.ordered_set import OrderedSet
from flatland.envs.rail_env import RailAgentStatus

import utils
from railway_encoding import CellOrientationGraph


'''
    -Graph:
        - Divide intersection nodes on arrival direction
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
            - [Y] Num. agents in deadlock in the previous path if the other agents follow their shortest path
            - [Y] Distance from the nearest deadlock in the path computed as above
            - [Y] Number of agent malfunctioning in the previous path in the same direction
            - [Y] Number of agent malfunctioning in the previous path in the opposite direction
            - [Y] Turns to wait blocked if there is an agent on path malfunctioning in the same direction (difference between malfunction time and distance)
            - [Y] Turns to wait blocked if there is an agent on path malfunctioning in the opposite direction (difference between malfunction time and distance)
        - Todo:
            - Test new Deadlock and deadlock distances
            - Packed paths and weights with numpy and only positions, remove direction
            - Add partial shortest path feature results to deviations path
            - Check normalization correctness
            - Differentiate `inf` from maximum value in normalization
            - Handle observation not present at all (all -1 or all 0 ???)
'''

SpeedData = namedtuple('SpeedData', ['times', 'remaining'])


class CustomObservation(ObservationBuilder):

    FEATURES = 12

    def __init__(self, max_depth, predictor):
        super().__init__()
        self.max_depth = max_depth
        self.predictor = predictor
        self.observations = dict()
        self.agent_handles = set()
        self.other_agents = dict()
        self.speed_data = dict()
        self.last_nodes = []

    def _init_env(self):
        self.railway_encoding = CellOrientationGraph(
            grid=self.env.rail.grid, agents=self.env.agents
        )
        if self.predictor is not None:
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
        self.last_nodes = [
            self.railway_encoding.previous_node(
                self.railway_encoding.get_agent_cell(handle)
            ) for handle in self.agent_handles
        ]

    def reset(self):
        self._init_env()
        self._init_agents()
        self._init_speed_data()

    def set_env(self, env):
        super().set_env(env)
        if self.predictor:
            self.predictor.set_env(self.env)

    def get_many(self, handles=None):
        self.predictions = self.predictor.get_many()
        self._shortest_paths = np.full(
            (len(self.agent_handles), self.max_depth),
            -1, np.dtype('int, int, int')
        )
        self._shortest_positions = np.full(
            (len(self.agent_handles), self.max_depth), -1, np.dtype('int, int')
        )
        self._shortest_cum_weights = np.full(
            (len(self.agent_handles), self.max_depth), np.inf
        )
        for handle, val in self.predictions.items():
            # Check if agent is not at Target
            if self.predictions[handle] is not None:

                # Update last visited node for each agent
                agent_position = self.railway_encoding.get_agent_cell(handle)
                if agent_position in self.railway_encoding.graph.nodes:
                    self.last_nodes[handle] = (agent_position, 0)
                else:
                    self.last_nodes[handle] = (
                        self.last_nodes[handle][0],
                        self.last_nodes[handle][1]+1
                    )

                remaining_steps = 0
                if self.env.agents[handle].speed_data["speed"] < 1.0:
                    remaining_steps = int(
                        (1 - np.clip(self.env.agents[handle].speed_data["position_fraction"], 0.0, 1.0)
                         ) / self.env.agents[handle].speed_data["speed"]
                    )
                self.speed_data[handle] = SpeedData(
                    times=self.speed_data[handle].times,
                    remaining=remaining_steps
                )
                self._update_data(handle, val[0], remaining_steps)

                #print(f'Handle: {handle} Deviation Paths:')
                # for dev in val[1]:
                #    print(dev.path)

        #print(f'SHORTEST PATHS: \n {self._shortest_paths} \n')
        #print(f'CUMULATIVE WEIGHTS: \n {self._shortest_cum_weights}\n')
        return super().get_many(handles)

    def _update_data(self, handle, prediction, remaining_steps):
        shortest_path = np.array(prediction.path, np.dtype('int, int, int'))
        shortest_positions = np.array(
            [node[:-1] for node in prediction.path], np.dtype('int, int')
        )
        shortest_cum_weights = np.array(
            self.compute_cumulative_weights(
                handle, prediction.edges, remaining_steps
            )
        )
        self._shortest_paths[handle, :shortest_path.shape[0]] = shortest_path
        self._shortest_positions[handle, :shortest_positions.shape[0]] = (
            shortest_positions
        )
        self._shortest_cum_weights[handle, :shortest_cum_weights.shape[0]] = (
            shortest_cum_weights
        )

    def get(self, handle=0):
        self.observations[handle] = np.full(
            (self.max_depth, self.max_depth, self.FEATURES), -np.inf
        )
        if self.predictions[handle] is not None:
            shortest_path_prediction, deviation_paths_prediction = self.predictions[handle]
            shortest_feats = self._fill_path_values(
                handle, shortest_path_prediction
            )
            self.observations[handle][0, :, :] = shortest_feats
            for i, deviation_prediction in enumerate(deviation_paths_prediction):
                dev_feats = self._fill_path_values(
                    handle, deviation_prediction,
                    remaining_steps=self._shortest_cum_weights[handle, i],
                    shortest=False
                )
                self.observations[handle][i + 1, :, :] = dev_feats

        '''
        print()
        print(f'PREHandle: {handle}')
        print(self.observations[handle])
        print()
        '''
        self.observations[handle] = self.normalize(self.observations[handle])
        '''
        print()
        print(f'POSTHandle: {handle}')
        print(self.observations[handle])
        print()
        '''
        return self.observations[handle]

    def _fill_path_values(self, handle, prediction, remaining_steps=0, shortest=True):
        '''
        '''
        packed_paths, packed_weights = self._get_shortest_packed_paths()
        path = prediction.path
        # Compute cumulative weights for the given path
        path_weights = np.zeros((self.max_depth,))
        weights = np.array(
            self.compute_cumulative_weights(
                handle, prediction.edges, remaining_steps
            )
        )
        path_weights[:weights.shape[0]] = weights

        # Compute features
        num_agents, agent_distances = self.agents_in_path(
            handle, path, path_weights
        )
        target_distances = self.distance_from_target(
            handle, prediction.lenght, path, path_weights, remaining_steps
        )
        c_nodes = self.common_nodes(handle, path)
        if shortest:
            path = packed_paths[handle].tolist()[:len(path)]
            path_weights = packed_weights[handle].tolist()

        #print(f'Handle {handle} -> Path {path}')
        deadlocks, deadlock_distances = self.find_deadlocks(
            handle, path, path_weights, packed_paths, packed_weights
        )
        # print(deadlocks)
        # print(deadlock_distances)
        # print()
        # Build the feature matrix
        feature_matrix = np.vstack([
            num_agents, agent_distances, target_distances,
            c_nodes, deadlocks, deadlock_distances
        ]).T

        return feature_matrix

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
        num_agents = np.zeros((self.max_depth, 4))
        distances = np.ones((self.max_depth, 4)) * np.inf

        # For each agent different than myself
        for agent in self.other_agents[handle]:
            directions = [0]
            position = self.railway_encoding.get_agent_cell(agent)
            # Check if agent is not DONE_REMOVED -> position is None
            if position is not None:
                node, remaining_distance = self.railway_encoding.next_node(
                    position
                )
                nodes = self.railway_encoding.get_nodes((node[0], node[1]))
                next_nodes = list(self.railway_encoding.graph.successors(node))
                for other_node in nodes:
                    index = utils.get_index(path, other_node)
                    if index:
                        if (other_node != node and
                            (len(next_nodes) > 1 or len(path) <= index + 1 or
                             (len(next_nodes) > 0 and next_nodes[0] != path[index + 1]))):
                            directions = [1]

                        malfunction = self.env.agents[agent].malfunction_data['malfunction']
                        if malfunction > 0:
                            directions.append(directions[0] + 2)
                        break
                if index:
                    # For each computed direction (same and opposite)
                    for direction in directions:
                        num_agents[index:len(path), direction] += 1
                        value = (
                            cum_weights[index] +
                            self.speed_data[agent].remaining
                        )
                        # Same direction
                        if direction % 2 == 0:
                            value -= (
                                remaining_distance *
                                self.speed_data[handle].times
                            )
                        # Other direction
                        else:
                            value += (
                                remaining_distance *
                                self.speed_data[handle].times
                            )
                        # If malfunctioning
                        if direction >= 2:
                            value = np.clip(malfunction - value, 0, None)
                        # Update distances
                        if ((direction < 2 and distances[index, direction] > value) or
                                (direction >= 2 and distances[index, direction] < value)):
                            distances[index:len(path), direction] = value
        return np.transpose(num_agents), np.transpose(distances)

    def distance_from_target(self, handle, lenght, path, cum_weights, remaining_steps=0):
        '''
        Given the full lenght of a path from a root node to an agent's target,
        compute the distance from each node of the path to the target
        '''
        distances = np.zeros((self.max_depth,))
        if lenght == np.inf:
            return np.full((self.max_depth,), np.inf)
        distances[:len(path)] = (
            np.ones((len(path),))
            * (lenght + remaining_steps)
            * self.speed_data[handle].times
        )
        distances = (
            (distances - cum_weights - remaining_steps)
            + 2 * self.speed_data[handle].remaining
        )
        return distances

    def common_nodes(self, handle, path):
        '''
        Given an agent's path and corresponding paths for every other agent,
        compute the number of intersections for each node
        '''
        c_nodes = np.zeros((self.max_depth,))
        if len(path) > 0:
            nd_path = np.array(
                [node[:-1] for node in path], np.dtype('int, int')
            )
            computed = np.zeros((len(path),))
            for row in self.other_agents[handle]:
                computed += np.count_nonzero(
                    np.isin(
                        nd_path, self._shortest_positions[row, :]
                    ).reshape(1, len(nd_path)),
                    axis=0
                )
            c_nodes[:computed.shape[0]] = computed
        return c_nodes

    def _get_shortest_packed_paths(self):
        # Compute Number of turn inside of current edge for each agent to find possible in-edge deadlock
        prev_weights = []
        prev_nodes = [node[0] for node in self.last_nodes]
        for i, p in enumerate(self._shortest_paths):
            if tuple(p[0]) != prev_nodes[i]:
                prev_weights.append(
                    - ((self.last_nodes[i][1] * self.speed_data[i].times) +
                       self.speed_data[i].times -
                       self.speed_data[i].remaining)
                )
            else:
                prev_weights.append(self._shortest_cum_weights[i, 0])

        packed_paths = np.hstack([
            np.array(prev_nodes, np.dtype('int, int, int')
                     ).reshape(self._shortest_paths.shape[0], 1),
            self._shortest_paths[:, 1:]
        ])
        packed_weights = np.hstack([
            (np.array(prev_weights) + self._shortest_cum_weights[:, 0]).reshape(
                self._shortest_cum_weights.shape[0], 1),
            self._shortest_cum_weights[:, 1:]
        ])
        return packed_paths, packed_weights

    def find_deadlocks(self, handle, path, cum_weights, packed_paths, packed_weights):
        deadlocks = np.zeros((self.max_depth,))
        crash_turns = np.ones((self.max_depth,)) * np.inf
        paths = packed_paths.tolist()
        for agent in self.other_agents[handle]:
            deadlock_found = False
            agent_path = packed_paths[agent]
            for i in range(len(agent_path) - 1):
                for j in range(len(path) - 1):
                    source = path[j]
                    dest = path[j+1]
                    if source[0] == agent_path[i+1][0] and source[1] == agent_path[i+1][1] and dest[0] == agent_path[i][0] and dest[1] == agent_path[i][1] and not cum_weights[j] > packed_weights[agent, i+1] and not cum_weights[j+1] < packed_weights[agent, i]:
                        deadlock_found = True
                        space = self.railway_encoding.graph.get_edge_data(
                            source, dest
                        )["weight"]
                        # print(
                        #    f'Handle {handle} -> Deadlock Edge: {source}-{dest} Space: {space} \n My Turns {cum_weights[j],cum_weights[j+1]} \n Oth Turns {packed_weights[agent, i],packed_weights[agent, i+1]}')
                        deadlocks[j:len(path)] += 1
                        if cum_weights[j] < 0 and packed_weights[agent, i] < 0:
                            space += (cum_weights[j] /
                                      self.speed_data[handle].times)
                            space += (packed_weights[agent, i] /
                                      self.speed_data[agent].times)
                        elif cum_weights[j] > packed_weights[agent, i]:
                            space -= abs(
                                cum_weights[j] -
                                abs(packed_weights[agent, i])
                            ) / self.speed_data[agent].times
                        elif packed_weights[agent, i] > cum_weights[j]:
                            space += abs(
                                packed_weights[agent, i] -
                                abs(cum_weights[j])
                            ) / self.speed_data[agent].times
                        crash_turn = np.ceil(
                            np.clip(cum_weights[j], 0, None) +
                            space / (1/self.speed_data[agent].times + 1 /
                                     self.speed_data[handle].times)
                        )
                        if crash_turns[j] > crash_turn:
                            crash_turns[j:len(path)] = crash_turn
                        break
                if deadlock_found:
                    break
        return deadlocks, crash_turns

    def normalize(self, observation):
        normalized_observation = np.full(
            (self.max_depth, self.max_depth, self.FEATURES), -1, np.dtype('float32')
        )
        num_agents = observation[:, :, 0:4]
        agent_distances = observation[:, :, 4:8]
        target_distances = observation[:, :, 8]
        c_nodes = observation[:, :, 9]
        deadlocks = observation[:, :, 10]
        deadlock_distances = observation[:, :, 11]

        # Normalize number of agents in path
        done_agents = sum([
            1 for i in self.agent_handles
            if self.env.agents[i].status in (RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED)
        ])
        remaining_agents = len(self.agent_handles) - done_agents
        num_agents /= remaining_agents

        # Normalize common nodes
        c_nodes /= self.max_depth

        # Normalize deadlocks
        deadlocks /= remaining_agents

        # Normalize distances
        finite_target_distances = target_distances[
            np.isfinite(target_distances)
        ]
        try:
            max_distance = finite_target_distances.max()
            min_distance = finite_target_distances.min()
            if max_distance != min_distance:
                agent_distances = (
                    (agent_distances - min_distance) /
                    (max_distance - min_distance)
                )
                target_distances = (
                    (target_distances - min_distance) /
                    (max_distance - min_distance)
                )
                deadlock_distances = (
                    (deadlock_distances - min_distance) /
                    (max_distance - min_distance)
                )
        except:
            pass
        agent_distances[agent_distances == np.inf] = 1
        target_distances[target_distances == np.inf] = 1
        deadlock_distances[deadlock_distances == np.inf] = 1

        # Build the normalized observation
        normalized_observation[:, :, 0:4] = num_agents
        normalized_observation[:, :, 4:8] = agent_distances
        normalized_observation[:, :, 8] = target_distances
        normalized_observation[:, :, 9] = c_nodes
        normalized_observation[:, :, 10] = deadlocks
        normalized_observation[:, :, 11] = deadlock_distances
        normalized_observation[normalized_observation == -np.inf] = -1
        return normalized_observation
