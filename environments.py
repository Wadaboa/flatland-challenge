from flatland.envs.agent_utils import EnvAgent
from observations import GraphObservator
import numpy as np

from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.rail_env import RailEnv

from railway_encoding import CellOrientationGraph
import obs_normalization


class RailEnvWrapper(RailEnv):

    def __init__(self, *args, normalize=True, **kwargs):
        super(RailEnvWrapper, self).__init__(*args, **kwargs)
        self.railway_encoding = None
        self.normalize = normalize
        self.state_size = self._get_state_size()

    def reset(self, regenerate_rail=True, regenerate_schedule=True, activate_agents=False,
              random_seed=None):
        '''
        reset(regenerate_rail, regenerate_schedule,
              activate_agents, random_seed)

        The method resets the rail environment

        Parameters
        ----------
        regenerate_rail : bool, optional
            regenerate the rails
        regenerate_schedule : bool, optional
            regenerate the schedule and the static agents
        activate_agents : bool, optional
            activate the agents
        random_seed : bool, optional
            random seed for environment

        Returns
        -------
        observation_dict: Dict
            Dictionary with an observation for each agent
        info_dict: Dict with agent specific information

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
        elif isinstance(self.obs_builder, GraphObservator):
            n_nodes = np.power(self.obs_builder.max_depth, 2)
        return n_features_per_node * n_nodes

    def _normalize_obs(self, obs):
        if self.normalize:
            for handle in obs:
                if obs[handle] is not None:
                    if isinstance(self.obs_builder, TreeObsForRailEnv):
                        obs[handle] = obs_normalization.normalize_tree_obs(
                            obs[handle], self.obs_builder.max_depth, observation_radius=10)
                    elif isinstance(self.obs_builder, GraphObservator):
                        obs[handle] = obs_normalization.normalize_graph_obs(obs[handle], self.railway_encoding.remaining_agents(
                        ), self.malfunction_generator.get_process_data().max_duration)
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
