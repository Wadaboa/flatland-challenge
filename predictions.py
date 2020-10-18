import numpy as np

from flatland.core.env_prediction_builder import PredictionBuilder
from flatland.envs.rail_env import RailAgentStatus
from flatland.utils.ordered_set import OrderedSet


class ShortestPathPredictor(PredictionBuilder):

    def __init__(self, max_depth=None):
        super().__init__(max_depth)

    def reset(self):
        self._shortest_paths = {agent.handle: [] for agent in self.env.agents}

    def get_shortest_path(self, handle):
        '''
        Keep a list of shortest paths for the given agent.
        At each time step, update the already compute paths and delete the ones
        which cannot be followed anymore.
        The returned shortest paths have the agent's position as the first element.
        '''
        position = self.railway_encoding.get_agent_cell(handle)
        node, _ = self.railway_encoding.next_node(position)
        chosen_path = None
        paths_to_delete = []
        for i, shortest_path in enumerate(self._shortest_paths[handle]):
            lenght, path = shortest_path
            # Delete divergent path
            if node != path[0] and node != path[1]:
                paths_to_delete = [i] + paths_to_delete
                continue

            # Update agent position
            if path[0] != position:
                lenght -= 1
            path[0] = position

            # If the agent is on a packed graph node, drop it
            if path[0] == path[1]:
                path = path[1:]

            # Agent arrived to target
            if lenght == 0:
                chosen_path = lenght, path
                break

            # Select this path if no other path has been previously selected
            if chosen_path is None:
                chosen_path = lenght, path

            # Update shortest path
            self._shortest_paths[handle][i] = lenght, path

        # Delete divergent paths
        for i in paths_to_delete:
            del self._shortest_paths[handle][i]

        # Compute shortest paths, if no path is already available
        if chosen_path is None:
            self._shortest_paths[handle] = self.railway_encoding.shortest_paths(
                handle
            )
            chosen_path = self._shortest_paths[handle][0]

        return chosen_path

    def get(self, handle=None):
        agents = self.env.agents
        if handle is not None:
            agents = [self.env.agents[handle]]

        prediction_dict = {}
        for agent in agents:
            if agent.status == RailAgentStatus.DONE_REMOVED or agent.status == RailAgentStatus.DONE:
                prediction_dict[agent.handle] = None
                continue

            # Consider agent speed
            position = self.railway_encoding.get_agent_cell(agent.handle)
            agent_speed = agent.speed_data["speed"]
            times_per_cell = int(np.reciprocal(agent_speed))

            # Get the shortest path
            lenght, path = self.get_shortest_path(agent.handle)
            edges = self.railway_encoding.edges_from_path(path)
            pos = self.railway_encoding.positions_from_path(path)

            # Edit weights to account for agent speed
            for edge in edges:
                edge[2]['distance'] = edge[2]['weight'] * times_per_cell

            # Edit positions to account for agent speed
            positions = [pos[0]]
            for position in pos[1:]:
                positions.extend([position] * times_per_cell)

            # Limit the number of returned positions
            prediction = lenght, edges, positions
            visited = OrderedSet()
            if self.max_depth is not None:
                prediction = lenght, edges, positions[:self.max_depth]
                visited.update(positions[:self.max_depth])
            else:
                visited.update(positions)

            # Update GUI
            self.env.dev_pred_dict[agent.handle] = visited

            prediction_dict[agent.handle] = prediction

        return prediction_dict

    def set_env(self, env):
        super().set_env(env)

    def set_railway_encoding(self, railway_encoding):
        self.railway_encoding = railway_encoding
