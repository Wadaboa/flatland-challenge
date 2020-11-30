import os

import numpy as np

from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.rail_env import RailEnv, RailEnvActions, RailAgentStatus
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.persistence import RailEnvPersister
from flatland.utils.rendertools import RenderTool, AgentRenderVariant
from flatland.core.grid.grid4_utils import get_new_position

import obs_normalization
from railway_encoding import CellOrientationGraph
from binary_tree_obs import BinaryTreeObservator
from graph_obs import GraphObservator


class RailEnvWrapper(RailEnv):

    def __init__(self, *args, normalize=True, **kwargs):
        super(RailEnvWrapper, self).__init__(*args, **kwargs)
        self.railway_encoding = None
        self.normalize = normalize
        self.state_size = self._get_state_size()

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

    def check_if_all_blocked(self):
        '''
        Checks whether all the agents are blocked (full deadlock situation)
        '''
        # First build a map of agents in each position
        location_has_agent = {}
        for agent in self.agents:
            if agent.status in [RailAgentStatus.ACTIVE, RailAgentStatus.DONE] and agent.position:
                location_has_agent[tuple(agent.position)] = 1

        # Looks for any agent that can still move
        for handle in self.get_agent_handles():
            agent = self.agents[handle]
            if agent.status == RailAgentStatus.READY_TO_DEPART:
                agent_virtual_position = agent.initial_position
            elif agent.status == RailAgentStatus.ACTIVE:
                agent_virtual_position = agent.position
            elif agent.status == RailAgentStatus.DONE:
                agent_virtual_position = agent.target
            else:
                continue

            possible_transitions = self.rail.get_transitions(
                *agent_virtual_position, agent.direction
            )
            orientation = agent.direction

            for branch_direction in [(orientation + i) % 4 for i in range(-1, 3)]:
                if possible_transitions[branch_direction]:
                    new_position = get_new_position(
                        agent_virtual_position, branch_direction
                    )

                    if new_position not in location_has_agent:
                        return False

        # Full deadlock
        return True

    def save(self, path):
        '''
        Save the given RailEnv environment as pickle
        '''
        filename = os.path.join(
            path, f"{self.width}x{self.height}-{self.random_seed}.pkl"
        )
        RailEnvPersister.save(self, filename)

    def get_renderer(self):
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

        if random_seed:
            self._seed(random_seed)

        optionals = {}
        if regenerate_rail or self.rail is None:

            rail, optionals = self._generate_rail()

            self.rail = rail
            self.height, self.width = self.rail.grid.shape

            # Do a new set_env call on the obs_builder to ensure
            # that obs_builder specific instantiations are made according to the
            # specifications of the current environment : like width, height, etc
            self.obs_builder.set_env(self)

        if optionals and 'distance_map' in optionals:
            self.distance_map.set(optionals['distance_map'])

        if regenerate_schedule or regenerate_rail or self.get_num_agents() == 0:
            agents_hints = None
            if optionals and 'agents_hints' in optionals:
                agents_hints = optionals['agents_hints']

            schedule = self.schedule_generator(self.rail, self.number_of_agents, agents_hints, self.num_resets,
                                               self.np_random)
            self.agents = EnvAgent.from_schedule(schedule)

            # Get max number of allowed time steps from schedule generator
            # Look at the specific schedule generator used to see where this number comes from
            self._max_episode_steps = schedule.max_episode_steps

        self.agent_positions = np.zeros(
            (self.height, self.width), dtype=int) - 1

        # Reset agents to initial
        self.reset_agents()

        for agent in self.agents:
            # Induce malfunctions
            if activate_agents:
                self.set_agent_active(agent)

            self._break_agent(agent)

            if agent.malfunction_data["malfunction"] > 0:
                agent.speed_data['transition_action_on_cellexit'] = RailEnvActions.DO_NOTHING

            # Fix agents that finished their malfunction
            self._fix_agent_after_malfunction(agent)

        self.num_resets += 1
        self._elapsed_steps = 0

        # TODO perhaps dones should be part of each agent.
        self.dones = dict.fromkeys(
            list(range(self.get_num_agents())) + ["__all__"], False)

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

        info_dict = {
            'action_required': {i: self.action_required(agent) for i, agent in enumerate(self.agents)},
            'malfunction': {
                i: agent.malfunction_data['malfunction'] for i, agent in enumerate(self.agents)
            },
            'speed': {i: agent.speed_data['speed'] for i, agent in enumerate(self.agents)},
            'status': {i: agent.status for i, agent in enumerate(self.agents)}
        }
        # Return the new observation vectors for each agent
        observation_dict = self._get_observations()
        return (self._normalize_obs(observation_dict), info_dict)

    def step(self, *args, **kwargs):
        obs, rewards, dones, info = super().step(*args, **kwargs)
        return (self._normalize_obs(obs), rewards, dones, info)

    def _get_state_size(self):
        n_features_per_node = self.obs_builder.observation_dim
        n_nodes = 0
        if isinstance(self.obs_builder, TreeObsForRailEnv):
            n_nodes = sum([np.power(4, i)
                           for i in range(self.obs_builder.max_depth + 1)]
                          )
        elif isinstance(self.obs_builder, BinaryTreeObservator):
            n_nodes = sum(2**i for i in range(self.obs_builder.max_depth))
        elif isinstance(self.obs_builder, GraphObservator):
            n_nodes = 1
        return n_features_per_node * n_nodes

    def _normalize_obs(self, obs):
        if not self.normalize:
            return obs

        for handle in obs:
            if obs[handle] is not None:
                if isinstance(self.obs_builder, TreeObsForRailEnv):
                    obs[handle] = obs_normalization.normalize_tree_obs(
                        obs[handle], self.obs_builder.max_depth, observation_radius=10)
        return obs

    def _generate_rail(self):
        if "__call__" in dir(self.rail_generator):
            return self.rail_generator(
                self.width, self.height, self.number_of_agents, self.num_resets, self.np_random)
        elif "generate" in dir(self.rail_generator):
            return self.rail_generator.generate(
                self.width, self.height, self.number_of_agents, self.num_resets, self.np_random)
        raise ValueError(
            "Could not invoke __call__ or generate on rail_generator")
