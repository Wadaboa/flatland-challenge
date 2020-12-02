import os

import numpy as np

from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.rail_env import RailEnv, RailEnvActions, RailAgentStatus
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.persistence import RailEnvPersister
from flatland.utils.rendertools import RenderTool, AgentRenderVariant
from flatland.core.grid.grid4_utils import get_new_position

import obs_normalization
from deadlocks import DeadlocksDetector
from railway_encoding import CellOrientationGraph
from binary_tree_obs import BinaryTreeObservator
from graph_obs import GraphObservator


class RailEnvWrapper(RailEnv):
    '''
    Railway environment wrapper, to handle custom logic
    '''

    def __init__(self, *args, normalize=True, **kwargs):
        super(RailEnvWrapper, self).__init__(*args, **kwargs)
        self.railway_encoding = None
        self.normalize = normalize
        self.state_size = self._get_state_size()
        self.deadlocks_detector = DeadlocksDetector()

    def _get_state_size(self):
        '''
        Compute the state size based on observation type
        '''
        n_features_per_node = self.obs_builder.observation_dim
        n_nodes = 0
        if isinstance(self.obs_builder, TreeObsForRailEnv):
            n_nodes = sum(
                4 ** i for i in range(self.obs_builder.max_depth + 1)
            )
        elif isinstance(self.obs_builder, BinaryTreeObservator):
            n_nodes = sum(2 ** i for i in range(self.obs_builder.max_depth))
        elif isinstance(self.obs_builder, GraphObservator):
            n_nodes = 1
        return n_features_per_node * n_nodes

    def get_agents_same_start(self):
        '''
        Return a dictionary indexed by agents starting positions,
        and having a list of handles as values, s.t. agents with
        the same starting position are ordered by decreasing speed
        '''
        agents_with_same_start = dict()
        for handle_one, agent_one in enumerate(self.agents):
            for handle_two, agent_two in enumerate(self.agents):
                if handle_one != handle_two and agent_one.initial_position == agent_two.initial_position:
                    agents_with_same_start.setdefault(
                        agent_one.initial_position, set()
                    ).update({handle_one, handle_two})

        for position in agents_with_same_start:
            agents_with_same_start[position] = sorted(
                list(agents_with_same_start[position]), reverse=True,
                key=lambda x: self.agents[x].speed_data['speed']
            )
        return agents_with_same_start

    def check_if_all_blocked(self, deadlocks):
        '''
        Checks whether all the agents are blocked (full deadlock situation)
        '''
        remaining_agents = self.railway_encoding.remaining_agents_handles()
        num_deadlocks = sum(
            int(v) for k, v in deadlocks.items()
            if k in remaining_agents
        )
        return num_deadlocks == len(remaining_agents)

    def save(self, path):
        '''
        Save the given RailEnv environment as pickle
        '''
        filename = os.path.join(
            path, f"{self.width}x{self.height}-{self.random_seed}.pkl"
        )
        RailEnvPersister.save(self, filename)

    def get_renderer(self):
        '''
        Return a renderer for the current environment
        '''
        return RenderTool(
            self,
            agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
            show_debug=True,
            screen_height=1080,
            screen_width=1920
        )

    def reset(self, regenerate_rail=True, regenerate_schedule=True,
              activate_agents=False, random_seed=None):
        '''
        Reset the environment
        '''
        # Get a random seed
        if random_seed:
            self._seed(random_seed)

        # Regenerate the rail, if necessary
        optionals = {}
        if regenerate_rail or self.rail is None:
            rail, optionals = self._generate_rail()
            self.rail = rail
            self.height, self.width = self.rail.grid.shape
            self.obs_builder.set_env(self)

        # Set the distance map
        if optionals and 'distance_map' in optionals:
            self.distance_map.set(optionals['distance_map'])

        # Regenerate the schedule, if necessary
        if regenerate_schedule or regenerate_rail or self.get_num_agents() == 0:
            agents_hints = None
            if optionals and 'agents_hints' in optionals:
                agents_hints = optionals['agents_hints']

            schedule = self.schedule_generator(
                self.rail, self.number_of_agents,
                agents_hints, self.num_resets, self.np_random
            )
            self.agents = EnvAgent.from_schedule(schedule)
            self._max_episode_steps = schedule.max_episode_steps

        # Reset agents positions
        self.agent_positions = np.full(
            (self.height, self.width), -1, dtype=int
        )
        self.reset_agents()
        for agent in self.agents:
            if activate_agents:
                self.set_agent_active(agent)
            self._break_agent(agent)
            if agent.malfunction_data["malfunction"] > 0:
                agent.speed_data['transition_action_on_cellexit'] = RailEnvActions.DO_NOTHING
            self._fix_agent_after_malfunction(agent)

        # Reset common variables
        self.num_resets += 1
        self._elapsed_steps = 0
        self.dones = dict.fromkeys(
            list(range(self.get_num_agents())) + ["__all__"], False
        )

        # Build the cell orientation graph
        self.railway_encoding = CellOrientationGraph(
            grid=self.rail.grid, agents=self.agents
        )

        # Reset the state of the observation builder with the new environment
        self.obs_builder.reset()
        self.distance_map.reset(self.agents, self.rail)

        # Reset the malfunction generator
        if "generate" in dir(self.malfunction_generator):
            self.malfunction_generator.generate(reset=True)
        else:
            self.malfunction_generator(reset=True)

        # Empty the episode store of agent positions
        self.cur_episode = []

        # Compute deadlocks
        self.deadlocks_detector.reset(self.get_num_agents())

        # Build the info dict
        info_dict = {
            'action_required': {}, 'malfunction': {}, 'speed': {},
            'status': {}, 'deadlocks': {}
        }
        for i, agent in enumerate(self.agents):
            info_dict['action_required'][i] = self.action_required(agent)
            info_dict['malfunction'][i] = agent.malfunction_data['malfunction']
            info_dict['speed'][i] = agent.speed_data['speed']
            info_dict['status'][i] = agent.status
            info_dict["deadlocks"][i] = self.deadlocks_detector.deadlocks[i]

        # Return the new observation vectors for each agent
        observation_dict = self._get_observations()
        return (self._normalize_obs(observation_dict), info_dict)

    def _generate_rail(self):
        '''
        Regenerate the rail, if necessary
        '''
        if "__call__" in dir(self.rail_generator):
            return self.rail_generator(
                self.width, self.height, self.number_of_agents, self.num_resets, self.np_random
            )
        elif "generate" in dir(self.rail_generator):
            return self.rail_generator.generate(
                self.width, self.height, self.number_of_agents, self.num_resets, self.np_random
            )
        raise ValueError(
            "Could not invoke __call__ or generate on rail_generator"
        )

    def step(self, *args, **kwargs):
        '''
        Perform a step in the environment
        '''
        obs, rewards, dones, info = super().step(*args, **kwargs)
        info["deadlocks"] = self.deadlocks_detector.step(self)
        return (self._normalize_obs(obs), rewards, dones, info)

    def _normalize_obs(self, obs):
        '''
        Normalize observations
        '''
        if not self.normalize:
            return obs

        for handle in obs:
            if obs[handle] is not None:
                # Normalize tree observation
                if isinstance(self.obs_builder, TreeObsForRailEnv):
                    obs[handle] = obs_normalization.normalize_tree_obs(
                        obs[handle], self.obs_builder.max_depth
                    )

        return obs
