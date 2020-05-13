import numpy as np

from flatland.core.grid.grid4_utils import get_new_position


class DistanceMapPredictor(PredictionBuilder):

    def __init__(self, max_depth=20):
        super().__init__(max_depth)

    def reset(self):
        '''
        Called after each environment reset
        '''
        pass

    def get(self, handle):
        '''
        Called whenever get_many in the observation build is called
        '''
        agent = self.env.agents[handle]

        if agent.position:
            possible_transitions = self.env.rail.get_transitions(
                *agent.position, agent.direction
            )
        else:
            possible_transitions = self.env.rail.get_transitions(
                *agent.initial_position, agent.direction
            )

        num_transitions = np.count_nonzero(possible_transitions)

        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right], relative to the current orientation
        # If only one transition is possible, the forward branch is aligned with it.
        if num_transitions == 1:
            observation = [0, 1, 0]
        else:
            min_distances = []
            for direction in [(agent.direction + i) % 4 for i in range(-1, 2)]:
                if possible_transitions[direction]:
                    new_position = get_new_position(agent.position, direction)
                    min_distances.append(
                        self.env.distance_map.get()[handle, new_position[0], new_position[1], direction])
                else:
                    min_distances.append(np.inf)

            observation = [0, 0, 0]
            min_distances = np.array(min_distances)
            min_indexes = np.where(min_distances == np.min(min_distances))
            for min_index in min_indexes:
                observation[min_index] = 1

        return observation


class ShortestPathPredictor(PredictionBuilder):

    def __init__(self, max_depth=20):
        super().__init__(max_depth)
        self._shortest_paths = {agent.handle: [] for agent in self.env.agents}

    def get_shortest_path(self, handle, position):
        if position in self._shortest_paths[handle]:
            return 

    def get(self, handle=None):
        agents = self.env.agents
        if handle:
            agents = [self.env.agents[handle]]

        prediction_dict = {}
        for agent in agents:

            if agent.status == RailAgentStatus.READY_TO_DEPART:
                agent_virtual_position = agent.initial_position
            elif agent.status == RailAgentStatus.ACTIVE:
                agent_virtual_position = agent.position
            elif agent.status == RailAgentStatus.DONE:
                agent_virtual_position = agent.target
            else:

                prediction = np.zeros(shape=(self.max_depth + 1, 5))
                for i in range(self.max_depth):
                    prediction[i] = [i, None, None, None, None]
                prediction_dict[agent.handle] = prediction
                continue

            agent_virtual_direction = agent.direction
            agent_speed = agent.speed_data["speed"]
            times_per_cell = int(np.reciprocal(agent_speed))
            prediction = np.zeros(shape=(self.max_depth + 1, 5))
            prediction[0] = [0, *agent_virtual_position,
                             agent_virtual_direction, 0]

            shortest_path = shortest_paths[agent.handle]

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
