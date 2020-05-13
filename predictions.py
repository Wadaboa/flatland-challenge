import numpy as np

from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.rail_env import RailEnvActions, RailAgentStatus


class ShortestPathPredictor(PredictionBuilder):

    def __init__(self, max_depth=None):
        super().__init__(max_depth)
        self._shortest_paths = {agent.handle: [] for agent in self.env.agents}

    def get_shortest_path(self, handle):
        position = self.railway_encoding.get_agent_cell(handle)
        node = self.railway_encoding.next_node(position)
        chosen_path = None
        for i, shortest_path in enumerate(self._shortest_paths[handle]):
            lenght, path = shortest_path
            if node == path[1]:
                lenght -= 1
                path[0] = position
                if node == position:
                    path = path[1:]
                if chosen_path is None:
                    chosen_path = path
            else:
                del self._shortest_paths[handle][i]

        if chosen_path is None:
            self._shortest_paths[handle] = self.railway_encoding.shortest_paths(
                handle
            )
            chosen_path = self._shortest_paths[handle][0]

        return chosen_path

    def get(self, handle=None):
        agents = self.env.agents
        if handle:
            agents = [self.env.agents[handle]]

        prediction_dict = {}
        for agent in agents:
            if agent.status == RailAgentStatus.DONE_REMOVED:
                prediction_dict[agent.handle] = None
                continue

            position = self.railway_encoding.get_agent_cell(agent.handle)
            agent_speed = agent.speed_data["speed"]
            times_per_cell = int(np.reciprocal(agent_speed))
            #prediction = np.zeros(shape=(self.max_depth + 1, 5))
            #prediction[0] = [0, position, 0]

            lenght, path = self.get_shortest_path(handle)
            edges = self.railway_encoding.edges_from_path(path)
            ##################################################

            # if there is a shortest path, remove the initial position
            if shortest_path:
                shortest_path = shortest_path[1:]

            new_direction = agent_virtual_direction
            new_position = agent_virtual_position
            visited = OrderedSet()
            for index in range(1, self.max_depth + 1):
                # if we're at the target, stop moving until max_depth is reached
                if new_position == agent.target or not shortest_path:
                    prediction[index] = [index, *new_position,
                                         new_direction, RailEnvActions.STOP_MOVING]
                    visited.add((*new_position, agent.direction))
                    continue

                if index % times_per_cell == 0:
                    new_position = shortest_path[0].position
                    new_direction = shortest_path[0].direction

                    shortest_path = shortest_path[1:]

                # prediction is ready
                prediction[index] = [index, *new_position, new_direction, 0]
                visited.add((*new_position, new_direction))

            # TODO: very bady side effects for visualization only: hand the dev_pred_dict back instead of setting on env!
            self.env.dev_pred_dict[agent.handle] = visited
            prediction_dict[agent.handle] = prediction

        return prediction_dict
