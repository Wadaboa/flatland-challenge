'''
'''


from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.grid4_utils import mirror, get_new_position
from flatland.envs.rail_env import RailEnvActions, RailAgentStatus

import networkx as nx
import matplotlib.pyplot as plt


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

        self.generate_graph()

    def generate_graph(self):
        edges = self.generate_edges()
        self._unpacked_graph = nx.DiGraph()
        self._unpacked_graph.add_edges_from(edges)
        nx.freeze(self._unpacked_graph)
        self.graph = nx.DiGraph(self._unpacked_graph)
        self.pack_graph()
        self.set_nodes_attributes()

    def pack_graph(self):
        '''
        Generate a compact version of the cell orientation graph,
        by only keeping junctions, targets and dead ends
        '''
        to_remove = self._straight_rails.difference(
            set(self._targets.keys())
        )
        for cell in to_remove:
            self.remove_cell(cell)

    def generate_edges(self):
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
                    for k, bit in enumerate(trans_bitmap):
                        if bit == '1':
                            original_dir, final_dir = self._BITMAP_TO_TRANS[k]
                            new_position_x, new_position_y = get_new_position(
                                [i, j], final_dir.value
                            )
                            edge = (
                                (i, j, original_dir.value),
                                (new_position_x, new_position_y, final_dir.value),
                                {
                                    'weight': 1,
                                    'action': self.agent_action(original_dir, final_dir)
                                }
                            )
                            edges.append(edge)
        return edges

    def agent_action(self, original_dir, final_dir):
        value = (final_dir.value - original_dir.value) % 4
        if value in (1, -3):
            return RailEnvActions.MOVE_RIGHT
        elif value in (-1, 3):
            return RailEnvActions.MOVE_LEFT
        return RailEnvActions.MOVE_FORWARD

    def set_nodes_attribute(self, positions, name, value, default=None):
        attributes = {}
        if default is not None:
            nx.set_node_attributes(self.graph, default, name)
        for pos in positions:
            nodes = self.get_nodes(pos)
            for node in nodes:
                val = value
                if isinstance(value, dict):
                    val = value[pos]
                attributes[node] = {name:  val}
        nx.set_node_attributes(self.graph, attributes)

    def set_nodes_attributes(self):
        self.set_nodes_attribute(
            self._dead_ends, 'is_dead_end', True, default=False
        )
        self.set_nodes_attribute(
            set(self._targets.keys()), 'is_target', True, default=False
        )
        self.set_nodes_attribute(
            set(self._targets.keys()), 'handles', self._targets
        )

    def remove_node(self, node):
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
                    'action': source[1]['action']
                }
            )
            for source in sources for target in targets
        ]
        self.graph.add_edges_from(new_edges)
        self.graph.remove_node(node)

    def remove_cell(self, position):
        nodes = self.get_nodes(position)
        for node in nodes:
            self.remove_node(node)

    def get_nodes(self, position):
        nodes = []
        for direction in TRANS:
            node = (position[0], position[1], direction.value)
            if self.graph.has_node(node):
                nodes.append(node)
        return nodes

    def next_node(self, cell):
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

    def get_agent_cell(self, handle):
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

    def shortest_paths(self, handle):
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
                if position != source:
                    path = [position] + path
                    lenght += weight
                paths.append((lenght, path))
            except nx.NetworkXNoPath:
                continue
        return sorted(paths, key=lambda x: x[0])

    def meaningful_subgraph(self, handle):
        nodes = {}
        source, _ = self.next_node(self.get_agent_cell(handle))
        for path in nx.all_simple_paths(self.graph, source, self.agents[handle].target):
            nodes.update(path)
        return nx.subgraph(self.graph, nodes)

    def edges_from_path(self, path):
        edges = []
        starting_index = 0
        if path[0] not in self.graph.nodes:
            fake_weight = nx.dijkstra_path_length(
                self._unpacked_graph, path[0], path[1]
            )
            edges.append((
                path[0], path[1],
                {'weight': fake_weight, 'action': RailEnvActions.MOVE_FORWARD}
            ))
            starting_index = 1
        for i in range(starting_index, len(path) - 1):
            edge = (path[i], path[i + 1])
            edge_attributes = self.graph.get_edge_data(*edge)
            edges.append((*edge, edge_attributes))
        return edges

    def positions_from_path(self, path):
        positions = []
        positions.append((path[0][0], path[0][1]))
        for i in range(0, len(path) - 1):
            _, mini_path = nx.bidirectional_dijkstra(
                self._unpacked_graph, path[i], path[i + 1]
            )
            mini_path = [(row, col) for row, col, _ in mini_path]
            positions.extend(mini_path[1:])
        return positions

    def draw_graph(self):
        nx.draw(self.graph, with_labels=True)
        plt.show()

    def draw_unpacked_graph(self):
        nx.draw(self._unpacked_graph, with_labels=True)
        plt.show()

    def draw_path(self, path):
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
