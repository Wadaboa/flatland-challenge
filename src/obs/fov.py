import torch
import numpy as np
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.data import Data

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.rail_env_grid import RailEnvTransitions

import utils


class FOVObservator(ObservationBuilder):
    '''
    An Observator that return a local observation of each agent in the form of
    a tensor of size max_depth centered around the agent position

    Features:
        0. Cell type of the rail in the agent's FOV
        1. Cell orientation of the rail in the agent's FOV
        2. Distances in shortest path in the agent's FOV
        3. Other agents positions in the agent's FOV (direction of each agent)
        4. Agents targets in the agent's FOV (1 agent target, 0 other agent, -1 otherwise)
        5. Agents malfunctioning turns
        6. Agents fractional speeds
        7+. Distances in deviation paths in the agent's FOV
    '''

    def __init__(self, max_depth, predictor):
        super().__init__()
        # Always keep an odd number of "squares", so that the agent
        # is centered w.r.t. its FOV
        assert max_depth % 2 != 0, 'FOV window must be an odd number'
        self.max_depth = max_depth
        self.predictor = predictor
        self.observations = dict()
        self.observation_dim = 7 + self.predictor.max_deviations
        self.possible_transitions_dict = self.compute_all_possible_transitions()
        self.agent_positions = None
        self.agent_malfunctions = None
        self.agent_speeds = None
        self.agent_targets = None
        self.data_object = None

    def reset(self):
        if self.predictor is not None:
            self.predictor.reset()
        rail_obs_16_channels = np.zeros((self.env.height, self.env.width, 16))
        for i in range(rail_obs_16_channels.shape[0]):
            for j in range(rail_obs_16_channels.shape[1]):
                bitlist = [
                    int(digit) for digit in
                    bin(self.env.rail.get_full_transitions(i, j))[2:]
                ]
                bitlist = [0] * (16 - len(bitlist)) + bitlist
                rail_obs_16_channels[i, j] = np.array(bitlist)
        self.rail_obs = self.convert_transitions_map(rail_obs_16_channels)
        self.agent_targets = np.full(
            (self.env.get_num_agents(), self.env.rail.height, self.env.rail.width), -1
        )
        for handle, agent in enumerate(self.env.agents):
            target = agent.target
            if target is not None:
                self.agent_targets[handle, target[0], target[1]] = 1
                other_agents = set(self.env.get_agent_handles()) - {handle}
                for other in other_agents:
                    if self.agent_targets[other, target[0], target[1]] == -1:
                        self.agent_targets[other, target[0], target[1]] = 0

    def set_env(self, env):
        super().set_env(env)
        if self.predictor:
            self.predictor.set_env(self.env)

    def convert_transitions_map(self, obs_transitions_map):
        '''
        Given an np.array of shape (env_height, env_width_, 16),
        convert it to (env_height,env_width, 2) where the first channel
        encodes cell types (empty cell 0 is encoded as -1,
        while cell types 1 to 10 are encoded as-is)
        and the second channel orientations (0, 90, 180, 270 as 0, 1, 2, 3)
        '''
        new_transitions_map = np.full(
            (obs_transitions_map.shape[0], obs_transitions_map.shape[1], 2), -1
        )

        for i in range(obs_transitions_map.shape[0]):
            for j in range(obs_transitions_map.shape[1]):
                transition_bitmap = obs_transitions_map[i, j]
                int_transition_bitmap = int(
                    transition_bitmap.dot(
                        2 ** np.arange(transition_bitmap.size)[::-1]
                    )
                )
                if int_transition_bitmap != 0:
                    new_transitions_map[i, j] = (
                        self.possible_transitions_dict[int_transition_bitmap]
                    )

        return new_transitions_map

    def compute_all_possible_transitions(self):
        '''
        Given transitions list considering cell types,
        outputs all possible transitions bitmap,
        considering cell rotations too
        '''
        # Bitmaps are read in decimal numbers
        transitions = RailEnvTransitions()
        transitions_with_rotation_dict = {}
        rotation_degrees = [0, 90, 180, 270]

        for index, transition in enumerate(transitions.transition_list):
            for rot_type, rot in enumerate(rotation_degrees):
                rot_transition = transitions.rotate_transition(transition, rot)
                if rot_transition not in transitions_with_rotation_dict:
                    transitions_with_rotation_dict[rot_transition] = (
                        np.array([index, rot_type])
                    )
        return transitions_with_rotation_dict

    def extract_path_fov(self, path, lenght, pad=0):
        '''
        Given a path returns the matrix fov marking the occupied positions assuming
        the first position as the center one of the fov
        '''
        path_fov = np.full((self.max_depth, self.max_depth), pad)
        distance = lenght
        if distance < np.inf:
            y, x = self.max_depth // 2, self.max_depth // 2
            prev_pos = path[0]
            for pos in path[1:]:
                if y >= 0 and y < self.max_depth and x >= 0 and x < self.max_depth:
                    path_fov[y, x] = distance
                    if pos[0] != prev_pos[0] or pos[1] != prev_pos[1]:
                        distance -= 1
                y += pos[0] - prev_pos[0]
                x += pos[1] - prev_pos[1]
                prev_pos = pos

            # Add last element
            if y >= 0 and y < self.max_depth and x >= 0 and x < self.max_depth:
                path_fov[y, x] = distance

        return path_fov

    def get_many(self, handles=None):
        self.agent_positions = np.full(
            (self.env.rail.height, self.env.rail.width), -1
        )
        self.agent_malfunctions = np.full(
            (self.env.rail.height, self.env.rail.width), -1
        )
        self.agent_speeds = np.full(
            (self.env.rail.height, self.env.rail.width), -1
        )
        if self.predictor is not None:
            self.predictions = self.predictor.get_many()
        for handle, agent in enumerate(self.env.agents):
            agent_position = self.env.railway_encoding.get_agent_cell(
                handle
            )
            if agent_position is not None:
                self.agent_positions[
                    agent_position[0], agent_position[1]
                ] = agent_position[2]
                self.agent_malfunctions[
                    agent_position[0], agent_position[1]
                ] = agent.malfunction_data['malfunction']
                self.agent_speeds[
                    agent_position[0], agent_position[1]
                ] = agent.speed_data['speed']

        # Compute adjacency matrix and store it in a
        # PyTorch Geometric Data object
        adjacency = self.env.agents_adjacency_matrix(
            radius=self.max_depth
        )
        edge_index = torch.from_numpy(
            np.argwhere(adjacency != 0)
        ).long().t().contiguous()
        edge_weight = torch.from_numpy(
            adjacency[np.nonzero(adjacency)]
        ).float()
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value=1, num_nodes=self.env.get_num_agents()
        )

        # Add features to PyTorch Geometric Data object
        states = super().get_many(handles)
        self.data_object = Data(
            edge_index=edge_index,
            edge_weight=edge_weight,
            num_nodes=self.env.get_num_agents(),
            states=torch.tensor(
                list(states.values()), dtype=torch.float
            )
        )
        self.observations = {
            handle: self.data_object for handle in self.env.get_agent_handles()
        }
        return self.observations

    def get(self, handle=0):
        self.observations[handle] = np.full(
            (self.observation_dim, self.max_depth, self.max_depth), -1
        )
        if (self.predictions[handle] is not None):
            agent_position = self.env.railway_encoding.get_agent_cell(
                handle
            )
            if agent_position is not None:
                shortest_pred, deviations_pred = self.predictions[handle]
                num_devs = self.predictor.max_deviations

                # Cell type of the rail in the agent's FOV
                cell_type = utils.extract_fov(
                    self.rail_obs[:, :, 0], agent_position, self.max_depth, -1
                )

                # Cell orientation of the rail in the agent's FOV
                cell_orientation = utils.extract_fov(
                    self.rail_obs[:, :, 1], agent_position, self.max_depth, -1
                )

                # Distances in shortest path in the agent's FOV
                path_fov = self.extract_path_fov(
                    shortest_pred.positions, shortest_pred.lenght, pad=-1
                )

                # Distances in deviation paths in the agent's FOV
                dev_paths_fov = np.full(
                    (num_devs, self.max_depth, self.max_depth), -1
                )
                for i, dev_pred in enumerate(deviations_pred):
                    dev_path_fov = np.full(
                        (self.max_depth, self.max_depth), -1
                    )
                    if dev_pred.lenght < np.inf:
                        source_index = shortest_pred.positions.index(
                            dev_pred.positions[0]
                        )
                        if source_index != -1:
                            dev_pos = (
                                shortest_pred.positions[:source_index] +
                                dev_pred.positions
                            )
                            dev_path_fov = self.extract_path_fov(
                                dev_pos, len(dev_pos), pad=-1
                            )
                    dev_paths_fov[i] = dev_path_fov

                # Other agents positions in the agent's FOV (direction of each agent)
                agents_fov = utils.extract_fov(
                    self.agent_positions, agent_position, self.max_depth, -1
                )

                # Agents targets in the agent's FOV (1 agent target, 0 other agent, -1 otherwise)
                targets_fov = utils.extract_fov(
                    self.agent_targets[handle, :, :],
                    agent_position, self.max_depth, -1
                )

                # Malfunctioning turns
                malf_fov = utils.extract_fov(
                    self.agent_malfunctions, agent_position, self.max_depth, -1
                )

                # Speed information
                speed_fov = utils.extract_fov(
                    self.agent_speeds, agent_position, self.max_depth, -1
                )

                # Update observations
                self.observations[handle][0] = cell_type
                self.observations[handle][1] = cell_orientation
                self.observations[handle][2] = path_fov
                self.observations[handle][3] = agents_fov
                self.observations[handle][4] = targets_fov
                self.observations[handle][5] = malf_fov
                self.observations[handle][6] = speed_fov
                self.observations[handle][7: 7 + num_devs + 1] = dev_paths_fov

        return self.observations[handle]
