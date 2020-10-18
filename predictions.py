import numpy as np

from flatland.core.env_prediction_builder import PredictionBuilder
from flatland.envs.rail_env import RailAgentStatus
from flatland.utils.ordered_set import OrderedSet


class ShortestPathPredictor(PredictionBuilder):

    def __init__(self, max_depth=None):
        super().__init__(max_depth)

    def reset(self):
        '''
        Initialize shortest paths for each agent
        '''
        self._shortest_paths = dict()
        for agent in self.env.agents:
            self._shortest_paths[agent.handle] = self.railway_encoding.shortest_paths(
                agent.handle
            )

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

    def get_many(self):
        '''
        Build the prediction for every agent
        '''
        prediction_dict = {}
        for agent in self.env.agents:
            prediction_dict[agent.handle] = self.get(agent.handle)
        return prediction_dict

    def get(self, handle):
        '''
        Build the prediction for the given agent
        '''
        agent = self.env.agents[handle]
        if agent.status == RailAgentStatus.DONE_REMOVED or agent.status == RailAgentStatus.DONE:
            return None

        # Consider agent speed
        position = self.railway_encoding.get_agent_cell(handle)
        agent_speed = agent.speed_data["speed"]
        times_per_cell = int(np.reciprocal(agent_speed))
        remaining_steps = int(
            (1 - agent.speed_data["position_fraction"]) / agent_speed
        )

        # Get the shortest path
        lenght, path = self.get_shortest_path(handle)
        edges = self.railway_encoding.edges_from_path(path)
        pos = self.railway_encoding.positions_from_path(path)

        # Edit weights to account for agent speed
        for edge in edges:
            edge[2]['distance'] = edge[2]['weight'] * times_per_cell
        edges[0][2]['distance'] -= (times_per_cell - remaining_steps)

        # Edit positions to account for agent speed
        positions = [pos[0]] * (remaining_steps)
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
        self.env.dev_pred_dict[handle] = visited

        return prediction

    def set_env(self, env):
        super().set_env(env)

    def set_railway_encoding(self, railway_encoding):
        self.railway_encoding = railway_encoding
