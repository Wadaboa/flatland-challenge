import torch
import numpy as np
from torch_geometric.data import Data

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.envs.agent_utils import RailAgentStatus

import env_utils


class GraphObservator(ObservationBuilder):

    def __init__(self, max_depth, predictor):
        super().__init__()
        self.max_depth = max_depth
        self.predictor = predictor
        self.observations = dict()
        self.observation_dim = 2

    def reset(self):
        if self.predictor is not None:
            self.predictor.reset()

    def set_env(self, env):
        super().set_env(env)
        if self.predictor:
            self.predictor.set_env(self.env)

    def get_many(self, handles=None):
        self.predictions = self.predictor.get_many()
        return super().get_many(handles)

    def get(self, handle=0):
        self.observations[handle] = self.get_graph_data(handle)
        return self.observations[handle]

    def get_graph_data(self, handle):
        # Compute edges and edges attributes
        edges = self.env.railway_encoding.get_graph_edges(
            unpacked=False, data=True
        )
        edge_index, edge_weight = [], []
        for u, v, d in edges:
            edge_index.append([
                self.env.railway_encoding.node_to_index[u],
                self.env.railway_encoding.node_to_index[v]
            ])
            edge_weight.append(d['weight'])
        edge_index = torch.tensor(
            edge_index, dtype=torch.long
        ).t().contiguous()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)

        # Compute node features
        nodes = self.env.railway_encoding.get_graph_nodes(
            unpacked=False, data=True
        )
        x = [None] * len(nodes)
        agents_positions = {
            self.env.railway_encoding.get_agent_cell(h)
            for h in range(len(self.env.agents)) if h != handle
        }
        for n, d in nodes:
            target_distance = self.env.distance_map.get()[
                handle, n[0], n[1], n[2]
            ]
            is_occupied = n in agents_positions
            x[self.env.railway_encoding.node_to_index[n]] = [
                #d["is_dead_end"], d["is_fork"], d["is_join"], d["is_target"],
                target_distance, is_occupied
            ]
        x = torch.tensor(x, dtype=torch.float)

        # Store a list of important positions, so that the DQN is called with
        # the GNN embeddings of these nodes
        agent_position = self.env.railway_encoding.get_agent_cell(handle)
        agent_pos_index = -1
        successors = []
        if agent_position is not None:
            agent_in_packed = self.env.railway_encoding.is_node(
                agent_position, unpacked=False
            )
            if agent_in_packed:
                successors = self.env.railway_encoding.get_successors(
                    agent_position, unpacked=False
                )
            else:
                actual_agent_position = tuple(agent_position)
                agent_position, _ = self.env.railway_encoding.previous_node(
                    actual_agent_position
                )
                successor, _ = self.env.railway_encoding.next_node(
                    actual_agent_position
                )
                successors = [successor]
            agent_pos_index = self.env.railway_encoding.node_to_index[agent_position]

        successors_indexes = {"left": -1, "right": -1}
        for succ in successors:
            succ_index = self.env.railway_encoding.node_to_index[succ]
            succ_choice = self.env.railway_encoding.get_edge_data(
                agent_position, succ, 'choice', unpacked=False
            )
            if succ_choice == env_utils.RailEnvChoices.CHOICE_LEFT:
                successors_indexes["left"] = succ_index
            elif succ_choice == env_utils.RailEnvChoices.CHOICE_RIGHT:
                successors_indexes["right"] = succ_index
        pos = torch.tensor([
            successors_indexes["left"],
            successors_indexes["right"],
            agent_pos_index
        ], dtype=torch.long)

        # Create a PyTorch Geometric Data object
        data = Data(
            edge_index=edge_index, edge_weight=edge_weight, pos=pos, x=x
        )
        return data


class MultiAgentGraphObservator(ObservationBuilder):
    """
    Gives a global observation of the entire rail environment.
    The observation is composed of the following elements:
        - obs_rail: array with dimensions (env.height, env.width, 2) with
            - first channel containing the cell types in [0, 10]
            - second channel containing the cell rotation [0, 90, 180, 270]
        - obs_agents_state: A 3D array (map_height, map_width, 5) with
            - first channel containing the agents position and direction
            - second channel containing the other agents positions and direction
            - third channel containing agent/other agent malfunctions
            - fourth channel containing agent/other agent fractional speeds
        - obs_targets: Two 2D arrays (map_height, map_width, 2) containing respectively the position of the given agent\
         target and the positions of the other agents targets (flag only, no counter!).
    """

    def __init__(self, max_depth, predictor):
        super().__init__()
        # Always keep an odd number of "squares", so that the agent
        # is centered w.r.t. its FOV
        self.max_depth = max_depth if max_depth % 2 != 0 else max_depth + 1
        self.predictor = predictor
        self.observations = dict()
        self.observation_dim = 2
        self.possible_transitions_dict = self.compute_all_possible_transitions()

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

    def set_env(self, env):
        super().set_env(env)
        if self.predictor:
            self.predictor.set_env(self.env)

    def convert_transitions_map(self, obs_transitions_map):
        '''
        Given np.array of shape (env_width_ env_height, 16) convert to (env_width, env_height, 2) where
        the first channel encodes cell_types (0,.. 10) and the second channel orientation (0, 90, 180, 270 as 0 1 2 3)
        '''
        new_transitions_map = np.zeros(
            (obs_transitions_map.shape[0], obs_transitions_map.shape[1], 2)
        )

        for i in range(obs_transitions_map.shape[0]):
            for j in range(obs_transitions_map.shape[1]):
                transition_bitmap = obs_transitions_map[i, j]
                int_transition_bitmap = int(
                    transition_bitmap.dot(
                        2 ** np.arange(transition_bitmap.size)[::-1]
                    )
                )
                new_transitions_map[i, j] = (
                    self.possible_transitions_dict[int_transition_bitmap]
                )

        return new_transitions_map

    def compute_all_possible_transitions(self):
        '''
        Given transitions list considering cell types outputs all possible transitions bitmap considering cell rotations too
        '''
        # Bitmaps are read in decimal numbers
        transitions = RailEnvTransitions()
        transition_list = transitions.transition_list
        transitions_with_rotation_dict = {}
        rotation_degrees = [0, 90, 180, 270]

        for i in range(len(transition_list)):
            for r in rotation_degrees:
                t = transition_list[i]
                rot_transition = transitions.rotate_transition(t, r)
                if rot_transition not in transitions_with_rotation_dict:
                    transitions_with_rotation_dict[rot_transition] = (
                        np.array([i, r])
                    )

        return transitions_with_rotation_dict

    def extract_fov(self, matrix, center_index, pad=0):
        '''
        Extract a patch from the given matrix centered around the specified position
        and pad external values with the given fill value
        '''
        # Window is entirely contained in the given matrix
        m, n = matrix.shape
        offset = self.max_depth // 2
        xl, xu = center_index[0] - offset, center_index[0] + offset
        yl, yu = center_index[1] - offset, center_index[1] + offset
        if xl >= 0 and xu < m and yl >= 0 and yu < n:
            return matrix[xl:xu + 1, yl: yu + 1]

        # Window has to be padded
        window = np.full((self.max_depth, self.max_depth), pad)
        c_xl, c_xu = np.clip(xl, 0, m), np.clip(xu, 0, m)
        c_yl, c_yu = np.clip(yl, 0, n), np.clip(yu, 0, n)
        sub = matrix[c_xl:c_xu + 1, c_yl: c_yu + 1]
        w_xl = 0 if xl >= 0 else abs(xl)
        w_xu = self.max_depth if xu < m else self.max_depth - (xu - m) - 1
        w_yl = 0 if yl >= 0 else abs(yl)
        w_yu = self.max_depth if yu < n else self.max_depth - (yu - n) - 1
        window[w_xl:w_xu, w_yl:w_yu] = sub
        return window

    def get(self, handle=0):
        obs = np.full(
            (self.max_depth, self.max_depth, self.observation_dim), -1
        )
        agent = self.env.agents[handle]
        agent_virtual_position = self.env.railway_encoding.get_agent_cell(
            handle
        )

        obs_targets = np.zeros((self.env.height, self.env.width, 2))
        obs_agents_state = np.zeros((self.env.height, self.env.width, 5)) - 1
        obs_agents_state[:, :, 4] = 0
        obs_agents_state[agent_virtual_position][0] = agent.direction
        obs_targets[agent.target][0] = 1

        for i in range(len(self.env.agents)):
            other_agent = self.env.agents[i]

            # ignore other agents not in the grid any more
            if other_agent.status == RailAgentStatus.DONE_REMOVED:
                continue

            obs_targets[other_agent.target][1] = 1

            # second to fourth channel only if in the grid
            if other_agent.position is not None:
                # second channel only for other agents
                if i != handle:
                    obs_agents_state[other_agent.position][1] = other_agent.direction
                obs_agents_state[other_agent.position][2] = other_agent.malfunction_data['malfunction']
                obs_agents_state[other_agent.position][3] = other_agent.speed_data['speed']
            # fifth channel: all ready to depart on this position
            if other_agent.status == RailAgentStatus.READY_TO_DEPART:
                obs_agents_state[other_agent.initial_position][4] += 1
        return self.rail_obs, obs_agents_state, obs_targets
