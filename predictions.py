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
