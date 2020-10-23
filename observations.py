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
            - [Y] Num. agents in deadlock in the previous path if the other agents follow their shortest path
            - Distance from the nearest deadlock in the path computed as above
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

    def set_env(self, env):
        super().set_env(env)
        if self.predictor:
            self.predictor.set_env(self.env)

    def get_many(self, handles=None):
        self.predictions = self.predictor.get_many()
        self._shortest_paths = np.full(
            (len(self.agent_handles), self.max_depth), -1, np.dtype('int, int, int'))
        self._shortest_pos = np.full(
            (len(self.agent_handles), self.max_depth), -1, np.dtype('int, int'))
        self._cumulative_weights = np.zeros(
            (len(self.agent_handles), self.max_depth))
        for handle, val in self.predictions.items():
            # Check if agent is not at Target
            if self.predictions[handle] is not None:
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
                # Check if exist a path
                if val[0].lenght < np.inf:
                    self._update_data(handle, val[0], remaining_steps)
        return super().get_many(handles)

    def _update_data(self, handle, prediction, remaining_steps):
        shortest_path = np.array(prediction.path, np.dtype('int, int, int'))
        shortest_pos = np.array([node[:-1]
                                 for node in prediction.path], np.dtype('int, int'))
        cum_weights = np.array(self.compute_cumulative_weights(
            handle, prediction.edges, remaining_steps))
        self._shortest_paths[handle, :shortest_path.shape[0]] = shortest_path
        self._shortest_pos[handle, :shortest_pos.shape[0]] = shortest_pos
        self._cumulative_weights[handle, :cum_weights.shape[0]] = cum_weights

    def get(self, handle=0):
        self.observations[handle] = np.ones(
            (self.max_depth, self.max_depth, self.FEATURES)
        ) * -np.inf
        if self.predictions[handle] is not None:

            shortest_path_prediction, deviation_paths_prediction = self.predictions[handle]

            num_agents, agent_distances = self.agents_in_path(
                handle,
                shortest_path_prediction.path,
                self._cumulative_weights[handle]
            )
            target_distances = self.distance_from_target(
                handle,
                shortest_path_prediction.lenght,
                shortest_path_prediction.path,
                self._cumulative_weights[handle]
            )
            c_nodes = self.common_nodes(
                handle, shortest_path_prediction.path, self._shortest_pos
            )
            deadlocks, deadlock_distances = self.find_deadlocks(
                handle, shortest_path_prediction.path, self._shortest_paths, self._cumulative_weights[handle], self._cumulative_weights)
            # Debug Zone
            if handle == 2:
                # Not Working -> See Comment Line 117
                # print(f'\nAgents in Path:\n Num agents\n {num_agents}\n Distances\n {agent_distances}\n')
                # Not Working -> See Comment Line 117
                # print(f'\nTarget Distances:\n Distances\n {target_distances}\n')
                # Working
                # print(f'\nCommon Nodes:\n Nodes\n {c_nodes}\n')
                # Not Working -> See Comment Line 117
                print(
                    f'\nDeadlock:\n Deadlock\n {deadlocks}\n Distances \n {deadlock_distances}\n')

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
        num_agents = np.zeros((self.max_depth, 4))
        distances = np.ones((self.max_depth, 4)) * np.inf
        for agent in self.other_agents[handle]:
            directions = [0]
            position = self.railway_encoding.get_agent_cell(agent)
            # Check if agent is not DONE_REMOVED -> position is None
            if position:
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
                        num_agents[index:len(path), direction] += 1
                        value = (cum_weights[index] +
                                 self.speed_data[agent].remaining)
                        if direction % 2 == 0:
                            value -= (
                                remaining_distance *
                                self.speed_data[handle].times
                            )
                        else:
                            value += (
                                remaining_distance *
                                self.speed_data[handle].times
                            )
                        if direction >= 2:
                            value = np.clip(malfunction - value, 0, None)
                        if ((direction < 2 and distances[index, direction] > value) or
                                (direction >= 2 and distances[index, direction] < value)):
                            distances[index:len(path), direction] = value
        return num_agents, distances

    def distance_from_target(self, handle, lenght, path, cum_weights):
        '''
        Given the full lenght of a path from a root node to an agent's target,
        compute the distance from each node of the path to the target
        '''
        distances = np.zeros((self.max_depth,))
        distances[:len(path)] = (
            np.ones((len(path),))
            * lenght
            * self.speed_data[handle].times
        )
        distances = (
            (distances - cum_weights)
            + 2 * self.speed_data[handle].remaining
        )
        return distances

    def common_nodes(self, handle, path, other_positions):
        '''
        Given an agent's path and corresponding paths for every other agent,
        compute the number of intersections for each node
        '''
        nd_path = np.array([node[:-1] for node in path], np.dtype('int, int'))
        c_nodes = np.zeros((self.max_depth,))
        computed = np.count_nonzero(
            np.isin(np.delete(other_positions, handle, axis=0), nd_path), axis=0)
        c_nodes[:computed.shape[0]] = computed
        return c_nodes

    def _common_edges(self, c_nodes, cum_weight, path):
        c_edges = []
        weights = []
        indexes = []
        for index in range(len(path)-1):
            if c_nodes[index] >= 1 and c_nodes[index+1] >= 1:
                c_edges.append((path[index], path[index+1]))
                weights.append((cum_weight[index], cum_weight[index+1]))
                indexes.append((index, index+1))
        return (c_edges, weights, indexes)

    def ffind_deadlocks(self, handle, c_nodes, cum_weight, path, other_paths, other_weights):
        deadlocks = np.zeros((self.max_depth,))
        distances = np.ones((self.max_depth,)) * np.inf
        if np.count_nonzero(c_nodes) > 0:
            edges, weights, indexes = self._common_edges(
                c_nodes, cum_weight, path)
            for edge, weight, index in zip(edges, weights, indexes):
                source, dest = edge
                space = self.railway_encoding.graph.get_edge_data(
                    source, dest)["weight"]
                my_speed = self.speed_data[handle].times
                w_sour, w_dest = weight
                ind_sour, ind_dest = index
                s_oth_dir = self.railway_encoding.get_nodes(
                    source[0], source[1])
                s_oth_dir.remove(source)
                d_oth_dir = self.railway_encoding.get_nodes(dest[0], dest[1])
                d_oth_dir.remove(dest)
                np_d_oth_dir = np.array(d_oth_dir, np.dtype('int,int,int'))
                np_s_oth_dir = np.array(s_oth_dir, np.dtype('int,int,int'))
                oth_edges = np.transpose([np.tile(np_d_oth_dir, len(
                    np_s_oth_dir)), np.repeat(np_s_oth_dir, len(np_d_oth_dir))])
                print(f'{source} - {dest} \n {oth_edges}')
                for i, row in enumerate(other_paths):
                    if i == handle:
                        continue
                    for oth_edge in oth_edges:
                        found_indexes = [x for x in range(
                            len(row)) if row[x:x+len(oth_edge)] == oth_edge]
                    if found_indexes:
                        if ((w_sour <= other_weights[i, found_indexes[0]] and
                             w_dest >= other_weights[i, found_indexes[1]]) or
                            (w_sour >= other_weights[i, found_indexes[0]]
                             and w_dest <= other_weights[i, found_indexes[1]])):
                            oth_speed = self.speed_data[i].times
                            deadlocks[ind_dest:len(path)] += 1
                            distance = int(w_sour + (
                                space - ((w_sour - other_weights[i, found_indexes[0]])/oth_speed)) * (oth_speed + my_speed))
                            if distances[ind_dest] > distance:
                                distances[ind_dest:len(path)] = distance
        return deadlocks, distances

    def find_deadlocks(self, handle, path, other_paths, cum_weights_path, cum_weights_other_path):
        '''
        Returns an array containing the number of deadlocks in the given path by looking
        at all possible future intersections with other agents' shortest paths
        '''
        padding = np.array((-1, -1, -1), np.dtype('int, int, int'))
        cum_weights_other_path = np.delete(
            cum_weights_other_path, handle, axis=0)
        already_found = []
        other_paths = np.delete(other_paths, handle, axis=0)
        deadlocks = np.zeros((self.max_depth,))
        distances = np.ones((self.max_depth,)) * np.inf
        init_weight = cum_weights_path[0]
        for index, weight in enumerate(cum_weights_path[1:]):
            traversed_indexes = np.logical_and(
                cum_weights_other_path >= init_weight,
                cum_weights_other_path <= weight
            )
            traversed_nodes = np.where(
                traversed_indexes, other_paths, padding)
            if handle == 2:
                print(traversed_nodes)
            for i, row in enumerate(traversed_nodes):
                useful_nodes = row[row != padding]
                if useful_nodes.size > 0 and index < len(path)-1:
                    last_node = useful_nodes[-1]
                    ln_index = np.argwhere(row == last_node)[0, 0]
                    good_index = None
                    if (i, ln_index) not in already_found:
                        if last_node[0] == path[index][0] and last_node[1] == path[index][1] and last_node[2] != path[index][2]:
                            good_index = index
                        elif last_node[0] == path[index+1][0] and last_node[1] == path[index+1][1] and last_node[2] != path[index+1][2]:
                            good_index = index + 1
                        if good_index:
                            already_found.append((i, ln_index))
                            deadlocks[good_index:len(path)] += 1
                            print(
                                f'{i} - {ln_index} - {cum_weights_other_path[i, ln_index]}')
                            value = (
                                2 * weight -
                                cum_weights_other_path[i, ln_index]
                            )
                            if distances[good_index] > value:
                                distances[good_index:len(path)] = value
        return deadlocks, distances
