'''
'''

import itertools
from collections import namedtuple

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.rail_env import RailEnvActions, RailAgentStatus

import env_utils


TRANS = [
    Grid4TransitionsEnum.NORTH,
    Grid4TransitionsEnum.EAST,
    Grid4TransitionsEnum.SOUTH,
    Grid4TransitionsEnum.WEST
]


class CellOrientationGraph():

    _BITMAP_TO_TRANS = [(t1, t2) for t1 in TRANS for t2 in TRANS]

    def __init__(self, grid, agents):
        self.grid = grid
        self.agents = agents
        self.graph = None
        self._unpacked_graph = None
        self._dead_ends = set()
        self._straight_rails = set()

        self._targets = dict()
        for agent in agents:
            self._targets.setdefault(agent.target, []).append(agent.handle)

        self._generate_graph()

        self.node_to_index, self.index_to_node = self.build_vocab(
            unpacked=False
        )

    def _generate_graph(self):
        '''
        Generate both the unpacked and the packed graph and
        set default attributes to the nodes in the packed graph
        '''
        edges = self._generate_edges()
        self._unpacked_graph = nx.DiGraph()
        self._unpacked_graph.add_edges_from(edges)
        nx.freeze(self._unpacked_graph)
        self.graph = nx.DiGraph(self._unpacked_graph)
        self._pack_graph()
        self._set_nodes_attributes()

    def _generate_edges(self):
        '''
        Translate the environment grid to the unpacked cell orientation graph
        '''
        edges = []
        for i, row in enumerate(self.grid):
            for j, _ in enumerate(row):
                if self.grid[i][j] != 0:
                    trans_int = self.grid[i][j]
                    trans_bitmap = format(trans_int, 'b').rjust(16, '0')
                    num_ones = trans_bitmap.count('1')
                    if num_ones == 2:
                        self._straight_rails.add((i, j))
                    elif num_ones == 1:
                        self._dead_ends.add((i, j))
                    tmp_edges, tmp_actions = [], dict()
                    for k, bit in enumerate(trans_bitmap):
                        if bit == '1':
                            original_dir, final_dir = self._BITMAP_TO_TRANS[k]
                            new_position_x, new_position_y = get_new_position(
                                [i, j], final_dir.value
                            )
                            tmp_action = env_utils.agent_action(
                                original_dir, final_dir
                            )
                            tmp_edges.append((
                                (i, j, original_dir.value),
                                (new_position_x, new_position_y, final_dir.value),
                                tmp_action
                            ))
                            tmp_actions.setdefault(
                                (i, j, original_dir.value), []
                            ).append(tmp_action)

                    for tmp_edge in tmp_edges:
                        tmp_choice = self.map_action_to_choice(
                            tmp_edge[2], tmp_actions[tmp_edge[0]]
                        )
                        edge = (
                            tmp_edge[0],
                            tmp_edge[1],
                            {
                                'weight': 1,
                                'action': tmp_edge[2],
                                'choice': tmp_choice
                            }
                        )
                        edges.append(edge)
        return edges

    def _pack_graph(self):
        '''
        Generate a compact version of the cell orientation graph,
        by only keeping junctions, targets and dead ends
        '''
        to_remove = self._straight_rails.difference(
            set(self._targets.keys())
        )
        for cell in to_remove:
            self._remove_cell(cell)

    def _remove_node(self, node):
        '''
        Remove a node from the in-construction packed graph and
        add an edge between the neighboring nodes, while
        also propagating edges data
        '''
        sources = [
            (source, data)
            for source, _, data in self.graph.in_edges(node, data=True)
        ]
        targets = [
            (target, data)
            for _, target, data in self.graph.out_edges(node, data=True)
        ]
        new_edges = [
            (
                source[0], target[0],
                {
                    'weight': source[1]['weight'] + target[1]['weight'],
                    'action': source[1]['action'],
                    'choice': source[1]['choice']
                }
            )
            for source in sources for target in targets
        ]
        self.graph.add_edges_from(new_edges)
        self.graph.remove_node(node)

    def _remove_cell(self, position):
        '''
        Remove the given cell with every direction component,
        in order to build the packed graph
        '''
        nodes = self.get_nodes(position)
        for node in nodes:
            self._remove_node(node)

    def _set_nodes_attribute(self, name, positions=None, value=None, default=None):
        '''
        Set the attribute "name" to the nodes given in the set "positions",
        to be "value" (could be a single value or a dictionary indexed by "positions").
        If the "value" argument is a dictionary, you can give a default value to be set
        to the nodes which are not present in the set "positions"
        '''
        if default is not None:
            nx.set_node_attributes(self.graph, default, name)
        attributes = {}
        if positions is not None and value is not None:
            for pos in positions:
                nodes = [pos]
                if len(pos) == 2:
                    nodes = self.get_nodes(pos)
                for node in nodes:
                    val = value
                    if isinstance(value, dict):
                        val = value[pos]
                    attributes[node] = {name:  val}
            nx.set_node_attributes(self.graph, attributes)

    def _set_nodes_attributes(self):
        '''
        Set default attributes for each and every node in the packed graph
        '''
        self._set_nodes_attribute(
            'is_dead_end', positions=self._dead_ends, value=True, default=False
        )
        self._set_nodes_attribute(
            'is_target', positions=set(self._targets.keys()), value=True, default=False
        )
        fork_positions, join_positions = self._compute_decision_types()
        self._set_nodes_attribute(
            'is_fork', positions=fork_positions, value=True, default=False
        )
        self._set_nodes_attribute(
            'is_join', positions=join_positions, value=True, default=False
        )

    def _compute_decision_types(self):
        '''
        Set decision types (at fork and/or at join) for each node in the packed graph
        '''
        fork_positions, join_positions = set(), set()
        for node in self.graph.nodes:
            if not self.graph.nodes[node]['is_dead_end']:
                other_nodes = set(self.get_nodes(node)) - {node}
                # If diamond crossing and/or fork set join for other nodes
                num_successors = len(self.get_successors(node))
                if len(other_nodes) == 3 or num_successors > 1:
                    for other_node in other_nodes:
                        join_positions.add(other_node)
                # Set fork for current node
                if num_successors > 1:
                    fork_positions.add(node)
        return fork_positions, join_positions

    def is_straight_rail(self, cell):
        '''
        Check if the given cell is a straight rail
        '''
        if len(cell) > 2:
            cell = cell[:-1]
        return cell in self._straight_rails

    def get_nodes(self, position, unpacked=False):
        '''
        Given a position (row, column), return a list
        of nodes present in the packed or unpacked graph of the type
        [(row, column, NORTH), ..., (row, column, WEST)]
        '''
        nodes = []
        for direction in TRANS:
            node = (position[0], position[1], direction.value)
            node_in_packed = not unpacked and self.graph.has_node(node)
            node_in_unpacked = unpacked and self._unpacked_graph.has_node(node)
            if node_in_packed or node_in_unpacked:
                nodes.append(node)
        return nodes

    def is_node(self, node, unpacked=False):
        '''
        Return true if the given node is present in the packed or
        unpacked graph
        '''
        graph = self._unpacked_graph if unpacked else self.graph
        return node in graph.nodes

    def get_edge_data(self, u, v, t, unpacked=False):
        graph = self.graph if not unpacked else self._unpacked_graph
        assert (u, v) in graph.edges
        edge_data = graph.get_edge_data(u, v)
        assert t in edge_data
        return edge_data[t]

    def get_predecessors(self, node, unpacked=False):
        '''
        Return the predecessors of the given node in the packed or
        unpacked graph
        '''
        graph = self._unpacked_graph if unpacked else self.graph
        if node not in graph.nodes:
            return []
        return list(graph.predecessors(node))

    def get_successors(self, node, unpacked=False):
        '''
        Return the successors of the given node in the packed or
        unpacked graph
        '''
        graph = self._unpacked_graph if unpacked else self.graph
        if node not in graph.nodes:
            return []
        return list(graph.successors(node))

    def next_node(self, cell):
        '''
        Return the closest node in the packed graph
        w.r.t. the given cell in the unpacked graph,
        in the same direction
        '''
        if cell in self.graph.nodes:
            return cell, 0
        weight = 0
        successors = self._unpacked_graph.successors(cell)
        while True:
            try:
                cell = next(successors)
                weight += 1
                if cell in self.graph.nodes:
                    return cell, weight
                successors = self._unpacked_graph.successors(cell)
            except StopIteration:
                break
        return None

    def previous_node(self, cell):
        '''
        Return the closest node in the packed graph
        w.r.t. the given cell in the unpacked graph,
        in the opposite direction
        '''
        if cell in self.graph.nodes:
            return cell, 0
        weight = 0
        next_node, _ = self.next_node(cell)
        predecessors = self._unpacked_graph.predecessors(cell)
        while True:
            try:
                cell = next(predecessors)
                weight += 1
                edge = (cell, next_node)
                if edge in self.graph.edges:
                    return cell, weight
                predecessors = itertools.chain(
                    predecessors, self._unpacked_graph.predecessors(cell)
                )
            except StopIteration:
                break
        return None

    def get_agent_cell(self, handle):
        '''
        Return the unpacked graph node in which the agent
        identified by the given handle is
        '''
        position = None
        agent = self.agents[handle]
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            position = (
                agent.initial_position[0],
                agent.initial_position[1],
                agent.initial_direction
            )
        elif agent.status == RailAgentStatus.ACTIVE:
            position = (
                agent.position[0],
                agent.position[1],
                agent.direction
            )
        elif agent.status == RailAgentStatus.DONE:
            position = (
                agent.target[0],
                agent.target[1],
                agent.direction
            )
        return position

    def is_done(self, handle):
        '''
        Returns True if an agent arrived at its target
        '''
        return self.agents[handle].status in (
            RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED
        )

    def map_choice_to_action(self, choice, actions):
        '''
        Map the given RailEnvChoices action to a RailEnvActions action
        '''
        # If CHOICE_LEFT, then priorities are MOVE_LEFT, MOVE_FORWARD, MOVE_RIGHT
        if choice == env_utils.RailEnvChoices.CHOICE_LEFT.value:
            if RailEnvActions.MOVE_LEFT in actions:
                return RailEnvActions.MOVE_LEFT.value
            elif RailEnvActions.MOVE_FORWARD in actions:
                return RailEnvActions.MOVE_FORWARD.value
            elif RailEnvActions.MOVE_RIGHT in actions:
                return RailEnvActions.MOVE_RIGHT.value
        # If CHOICE_RIGHT, then priorities are MOVE_RIGHT, MOVE_FORWARD
        elif choice == env_utils.RailEnvChoices.CHOICE_RIGHT.value:
            if RailEnvActions.MOVE_RIGHT in actions:
                return RailEnvActions.MOVE_RIGHT.value
            elif RailEnvActions.MOVE_FORWARD in actions:
                return RailEnvActions.MOVE_FORWARD.value
        # If STOP, then the priority is STOP_MOVING
        elif choice == env_utils.RailEnvChoices.STOP.value:
            return RailEnvActions.STOP_MOVING.value
        # Otherwise, last resort is DO_NOTHING
        return RailEnvActions.DO_NOTHING.value

    def map_action_to_choice(self, action, actions):
        if action == RailEnvActions.MOVE_LEFT and RailEnvActions.MOVE_LEFT in actions:
            return env_utils.RailEnvChoices.CHOICE_LEFT
        if action == RailEnvActions.MOVE_RIGHT and RailEnvActions.MOVE_RIGHT in actions:
            if len(actions) > 1:
                return env_utils.RailEnvChoices.CHOICE_RIGHT
            elif len(actions) == 1:
                return env_utils.RailEnvChoices.CHOICE_LEFT
        if action == RailEnvActions.MOVE_FORWARD and RailEnvActions.MOVE_FORWARD in actions:
            if RailEnvActions.MOVE_LEFT in actions:
                return env_utils.RailEnvChoices.CHOICE_RIGHT
            if RailEnvActions.MOVE_RIGHT in actions:
                return env_utils.RailEnvChoices.CHOICE_LEFT
            return env_utils.RailEnvChoices.CHOICE_LEFT
        return env_utils.RailEnvChoices.STOP

    def get_possible_choices(self, position, actions):
        # If only one agent, stop moving is not legal
        possible_moves = {
            env_utils.RailEnvChoices.CHOICE_LEFT: False,
            env_utils.RailEnvChoices.CHOICE_RIGHT: False,
            env_utils.RailEnvChoices.STOP: (
                self.is_before_join(position) and not self.only_one_agent()
            )
        }
        if RailEnvActions.MOVE_FORWARD in actions:
            # If RailEnvActions.MOVE_LEFT or RailEnvActions.MOVE_RIGHT in legal actions
            if len(actions) > 1:
                possible_moves[env_utils.RailEnvChoices.CHOICE_RIGHT] = True
            possible_moves[env_utils.RailEnvChoices.CHOICE_LEFT] = True
        if RailEnvActions.MOVE_LEFT in actions:
            possible_moves[env_utils.RailEnvChoices.CHOICE_LEFT] = True
        if RailEnvActions.MOVE_RIGHT in actions:
            # If only RailEnvActions.MOVE_RIGHT in legal actions
            if len(actions) == 1:
                possible_moves[env_utils.RailEnvChoices.CHOICE_LEFT] = True
            else:
                possible_moves[env_utils.RailEnvChoices.CHOICE_RIGHT] = True
        return list(possible_moves.values())

    def get_legal_choices(self, handle, actions):
        '''
        Map the given RailEnvActions actions to a list of RailEnvChoices
        '''
        # If the agent is arrived, only stop moving is possible
        # (necessary because of flatland bug)
        if self.is_done(handle):
            return [False, False, True]

        return self.get_possible_choices(self.get_agent_cell(handle), actions)

    def get_legal_actions(self, handle):
        '''
        Compute and return all the active legal actions that the given agent can perform
        (an active action is MOVE_*), by removing actions that lead into an occupied cell
        '''
        actions = self.get_agent_actions(handle)
        next_positions = [
            self.agent_position_by_action(handle, action)[:-1] for action in actions
        ]
        agents_positions = [
            self.get_agent_cell(agent)[:-1] for agent in range(len(self.agents))
            if (
                handle != agent and
                self.agents[agent].status not in
                (RailAgentStatus.DONE_REMOVED, RailAgentStatus.READY_TO_DEPART)
            )
        ]

        # Store illegal actions
        to_delete = []
        for i, pos in enumerate(next_positions):
            if pos is not None and pos in agents_positions:
                to_delete.append(i)

        # Delete illegal actions in reverse order,
        # so as to avoid changing indexes
        for i in sorted(to_delete, reverse=True):
            del actions[i]
        return actions

    def is_fork(self, position):
        if position in self.graph.nodes:
            return self.graph.nodes[position]['is_fork']
        return False

    def is_join(self, position):
        if position in self.graph.nodes and self.graph.nodes[position]['is_join']:
            return True
        return False

    def is_before_join(self, position):
        successors = self.get_successors(position, unpacked=True)
        for succ in successors:
            if self.is_join(succ):
                return True
        return False

    def is_at_fork(self, handle):
        '''
        Returns True iff the agent is at a fork
        '''
        return self.is_fork(self.get_agent_cell(handle))

    def is_at_before_join(self, handle):
        '''
        Returns True iff the agent is before a join
        '''
        return self.is_before_join(self.get_agent_cell(handle))

    def remaining_agents(self):
        done_agents = sum([
            self.is_done(agent) for agent in range(len(self.agents))
        ])
        return len(self.agents) - done_agents

    def only_one_agent(self):
        '''
        Returns True iff only one agent remains in the railway
        '''
        return self.remaining_agents() < 2

    def is_real_decision(self, handle):
        '''
        Returns True iff the agent has to make a decision
        '''
        return self.is_at_fork(handle) or (
            self.is_at_before_join(handle) and not self.only_one_agent()
        )

    def get_actions(self, position):
        '''
        Return all the possible active actions that can be performed from a given position
        (an active action is MOVE_*)
        '''
        successors = self.get_successors(position, unpacked=True)
        actions = []
        for succ in successors:
            actions.append(
                self._unpacked_graph.get_edge_data(position, succ)['action']
            )
        return actions

    def get_agent_actions(self, handle):
        '''
        Return all the possible active actions that an agent can perform
        (an active action is MOVE_*)
        '''
        return self.get_actions(self.get_agent_cell(handle))

    def action_from_positions(self, source, dest, unpacked=True):
        graph = self._unpacked_graph if unpacked else self.graph
        if (source, dest) in graph.edges:
            return graph.get_edge_data(source, dest)['action']
        return None

    def position_by_action(self, position, action):
        '''
        Return the next node if the given action will be performed in the given position
        '''
        successors = self.get_successors(position, unpacked=True)
        for succ in successors:
            if self._unpacked_graph.get_edge_data(position, succ)['action'] == action:
                return succ
        return None

    def agent_position_by_action(self, handle, action):
        '''
        Return the next node that the agent will occupy if it performs the given action
        '''
        self.position_by_action(self.get_agent_cell(handle), action)

    def shortest_paths(self, handle):
        '''
        Compute the shortest paths from the current position and direction,
        to the target of the agent identified by the given handle,
        considering every possibile target arrival direction.
        The shortest paths are then ordered by increasing lenght
        '''
        agent = self.agents[handle]
        position = self.get_agent_cell(handle)
        source, weight = self.next_node(position)
        targets = self.get_nodes(agent.target)
        paths = []
        for target in targets:
            try:
                lenght, path = nx.bidirectional_dijkstra(
                    self.graph, source, target
                )
                if position != path[0]:
                    path = [position] + path
                    lenght += weight
                paths.append((lenght, path))
            except nx.NetworkXNoPath:
                continue
        if not paths:
            return []
        return sorted(paths, key=lambda x: x[0])

    def deviation_paths(self, handle, source, node_to_avoid):
        '''
        Return alternative paths from `source` to the agent's target,
        without considering the actual shortest path
        '''
        agent = self.agents[handle]
        targets = self.get_nodes(agent.target)
        paths = []
        for succ in self.graph.successors(source):
            if succ != node_to_avoid:
                edge = self.graph.edges[(source, succ)]
                weight = edge['weight']
                for target in targets:
                    try:
                        lenght, path = nx.bidirectional_dijkstra(
                            self.graph, succ, target
                        )
                        path = [source] + path
                        lenght += weight
                        paths.append((lenght, path))
                    except nx.NetworkXNoPath:
                        continue
        if len(paths) == 0:
            return []
        return sorted(paths, key=lambda x: x[0])

    def meaningful_subgraph(self, handle):
        '''
        Return the subgraph which could be visited by the agent
        identified by the given handle
        '''
        nodes = {}
        source, _ = self.next_node(self.get_agent_cell(handle))
        for path in nx.all_simple_paths(self.graph, source, self.agents[handle].target):
            nodes.update(path)
        return nx.subgraph(self.graph, nodes)

    def get_distance(self, source, dest):
        '''
        Return the minimum distance between the source
        and destination nodes
        '''
        if (source not in self._unpacked_graph.nodes or
                dest not in self._unpacked_graph.nodes):
            return np.inf
        return nx.dijkstra_path_length(
            self._unpacked_graph, source, dest
        )

    def get_adjacency_matrix(self, unpacked=False):
        graph = self.graph if not unpacked else self._unpacked_graph
        return graph.to_scipy_sparse_matrix(
            dtype=np.dtype('long'), weight='weight', format='coo'
        )

    def get_graph_edges(self, unpacked=False, data=False):
        graph = self.graph if not unpacked else self._unpacked_graph
        return graph.edges(data=data)

    def get_graph_nodes(self, unpacked=False, data=False):
        graph = self.graph if not unpacked else self._unpacked_graph
        return graph.nodes(data=data)

    def build_vocab(self, unpacked=False):
        graph = self.graph if not unpacked else self._unpacked_graph
        nodes = sorted(list(graph.nodes()))
        node_to_index = {node: i for i, node in enumerate(nodes)}
        index_to_node = {i: node for i, node in enumerate(nodes)}
        return node_to_index, index_to_node

    def edges_from_path(self, path):
        '''
        Given a path in the packed graph as a sequence of nodes,
        return the corresponding sequence of edges
        '''
        edges = []
        starting_index = 0
        if path[0] not in self.graph.nodes:
            fake_weight, mini_path = nx.bidirectional_dijkstra(
                self._unpacked_graph, path[0], path[1]
            )
            edges.append((
                path[0], path[1],
                {
                    'weight': fake_weight,
                    'action': self._unpacked_graph.get_edge_data(mini_path[0], mini_path[1])['action'],
                    'choice': env_utils.RailEnvChoices.CHOICE_LEFT
                }
            ))
            starting_index = 1
        for i in range(starting_index, len(path) - 1):
            if path[i] != path[i + 1]:
                edge = (path[i], path[i + 1])
                edge_attributes = self.graph.get_edge_data(*edge)
                edges.append((*edge, edge_attributes))
        return edges

    def positions_from_path(self, path, max_lenght=None):
        '''
        Given a path in the packed graph, return the corresponding
        path in the unpacked graph, without the direction component
        '''
        positions = [path[0]]
        for i in range(0, len(path) - 1):
            _, mini_path = nx.bidirectional_dijkstra(
                self._unpacked_graph, path[i], path[i + 1]
            )
            positions.extend(mini_path[1:])
            if max_lenght is not None and len(positions) >= max_lenght:
                return positions[:max_lenght]
        return positions

    def different_direction_nodes(self, node):
        '''
        Given a node, described by row, column and direction,
        return every other node in the packed graph with
        a different direction component
        '''
        nodes = []
        row, col, direction = node
        for new_direction in range(len(TRANS)):
            new_node = (row, col, new_direction)
            if new_node != node and new_node in self.graph:
                nodes.append(new_node)
        return nodes

    def no_successors_nodes(self, unpacked=False):
        graph = self._unpacked_graph if unpacked else self.graph
        no_succ = []
        for node in graph.nodes:
            succ = self.get_successors(node, unpacked=unpacked)
            if len(succ) == 0:
                no_succ.append(node)
        return no_succ

    def draw_graph(self):
        '''
        Show the packed graph, with labels on nodes
        '''
        nx.draw(self.graph, with_labels=True)
        plt.show()

    def draw_unpacked_graph(self):
        '''
        Show the unpacked graph, with labels on nodes
        '''
        nx.draw(self._unpacked_graph, with_labels=True)
        plt.show()

    def draw_path(self, path):
        '''
        Show a path in the packed graph, where edges belonging
        to the path are colored in red
        '''
        if path[0] not in self.graph.nodes:
            path = path[1:]
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos)
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_nodes(self.graph, pos, nodelist=path, node_color='r')
        nx.draw_networkx_edges(
            self.graph, pos, edgelist=path_edges, edge_color='r', width=5
        )
        plt.axis('equal')
        plt.show()
