from collections import namedtuple

import numpy as np

from flatland.core.env_prediction_builder import PredictionBuilder
from flatland.envs.rail_env import RailAgentStatus
from flatland.utils.ordered_set import OrderedSet


Prediction = namedtuple('Prediction', ['lenght', 'path', 'edges', 'positions'])


def _empty_prediction():
    '''
    Return an empty Prediction namedtuple
    '''
    return Prediction(
        lenght=np.inf, path=[], edges=[], positions=[]
    )


class NullPredictor(PredictionBuilder):

    def __init__(self, max_depth=None):
        super().__init__(max_depth)

    def set_env(self, env):
        super().set_env(env)

    def get_many(self):
        '''
        Build the prediction for every agent
        '''
        return {agent.handle: None for agent in self.env.agents}

    def get(self, handle):
        '''
        Build the prediction for the given agent
        '''
        return None


class ShortestDeviationPathPredictor(PredictionBuilder):

    def __init__(self, max_depth, max_deviations):
        super().__init__(max_depth)
        self.max_deviations = max_deviations

    def set_env(self, env):
        super().set_env(env)

    def reset(self):
        '''
        Initialize shortest paths for each agent
        '''
        self._shortest_paths = dict()
        for agent in self.env.agents:
            self._shortest_paths[agent.handle] = self.env.railway_encoding.shortest_paths(
                agent.handle
            )

    def get_shortest_path(self, handle):
        '''
        Keep a list of shortest paths for the given agent.
        At each time step, update the already compute paths and delete the ones
        which cannot be followed anymore.
        The returned shortest paths have the agent's position as the first element.
        '''
        position = self.env.railway_encoding.get_agent_cell(handle)
        node, _ = self.env.railway_encoding.next_node(position)
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
            self._shortest_paths[handle] = self.env.railway_encoding.shortest_paths(
                handle
            )
            if not self._shortest_paths[handle]:
                if position == node:
                    node = self.env.railway_encoding.get_successors(node)[0]
                return np.inf, [position, node]

            chosen_path = self._shortest_paths[handle][0]

        return chosen_path

    def get_deviation_paths(self, handle, lenght, path):
        '''
        Return one deviation path for at most `max_deviations` nodes in the given path
        and limit the computed path lenghts by `max_depth`
        '''
        start = 0
        depth = min(self.max_deviations, len(path) - 1)
        deviation_paths = []
        padding = self.max_deviations
        if lenght < np.inf:
            padding -= len(path)
            source, _ = self.env.railway_encoding.next_node(path[0])
            if source != path[0]:
                start = 1
                deviation_paths.append(_empty_prediction())
            for i in range(start, depth):
                paths = self.env.railway_encoding.deviation_paths(
                    handle, path[i], path[i + 1]
                )
                deviation_path = []
                deviation_lenght = 0
                if len(paths) > 0:
                    deviation_path = paths[0][1]
                    deviation_lenght = paths[0][0]
                    edges = self.env.railway_encoding.edges_from_path(
                        deviation_path[:self.max_depth]
                    )
                    pos = self.env.railway_encoding.positions_from_path(
                        deviation_path[:self.max_depth]
                    )
                    deviation_paths.append(
                        Prediction(
                            lenght=deviation_lenght,
                            path=deviation_path[:self.max_depth],
                            edges=edges,
                            positions=pos
                        )
                    )
                else:
                    deviation_paths.append(_empty_prediction())

        deviation_paths.extend(
            [_empty_prediction()] * (padding)
        )
        return deviation_paths

    def get_many(self):
        '''
        Build the prediction for every agent
        '''
        prediction_dict = {}
        for agent in self.env.agents:
            prediction_dict[agent.handle] = None
            if agent.malfunction_data["malfunction"] == 0:
                prediction_dict[agent.handle] = self.get(agent.handle)
        return prediction_dict

    def get(self, handle):
        '''
        Build the prediction for the given agent
        '''
        agent = self.env.agents[handle]
        if agent.status == RailAgentStatus.DONE_REMOVED or agent.status == RailAgentStatus.DONE:
            return None

        # Build predictions
        lenght, path = self.get_shortest_path(handle)
        edges = self.env.railway_encoding.edges_from_path(
            path[:self.max_depth]
        )
        pos = self.env.railway_encoding.positions_from_path(
            path[:self.max_depth]
        )
        shortest_path_prediction = Prediction(
            lenght=lenght, path=path[:self.max_depth], edges=edges, positions=pos
        )
        deviation_paths_prediction = self.get_deviation_paths(
            handle, lenght, path
        )

        # Update GUI
        visited = OrderedSet()
        visited.update(shortest_path_prediction.positions)
        self.env.dev_pred_dict[handle] = visited

        return (shortest_path_prediction, deviation_paths_prediction)
