'''
'''


from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.grid4_utils import mirror, get_new_position
import networkx as nx


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
        self._unpacked_graph = None
        self.graph = None
        self._dead_ends = set()
        self._straight_rails = set()

        self._targets = dict()
        for agent in agents:
            self._targets.setdefault(agent.target, []).append(agent.handle)

        self.generate_graph()

    def generate_graph(self):
        edges = self.generate_edges()
        self._unpacked_graph = nx.DiGraph()
        self._unpacked_graph.add_weighted_edges_from(edges)
        nx.freeze(self._unpacked_graph)
        self.graph = nx.DiGraph(self._unpacked_graph)
        self.pack_graph()
        self.set_nodes_attributes()

    def pack_graph(self):
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
                            old_position_x, old_position_y = get_new_position(
                                [i, j], mirror(original_dir.value)
                            )
                            edge = (
                                (old_position_x, old_position_y, original_dir.value),
                                (i, j, final_dir.value), 1
                            )
                            edges.append(edge)
        return edges

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
            (source[0], target[0], source[1]['weight'] + target[1]['weight'])
            for source in sources for target in targets
        ]
        self.graph.add_weighted_edges_from(new_edges)
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
        print(self._unpacked_graph)
        if cell in self.graph.nodes:
            return cell, 0
        weight = 0
        successors = self._unpacked_graph.successors(cell)
        while True:
            print(successors)
            try:
                cell = next(successors)
                weight += 1
                if cell in self.graph.nodes:
                    return cell, weight
                successors = self._unpacked_graph.successors(cell)
            except StopIteration:
                break
        return None
