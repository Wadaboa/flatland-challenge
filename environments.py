import os
import copy

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

    def __init__(self, params, *args, normalize=True, **kwargs):
        super(RailEnvWrapper, self).__init__(*args, **kwargs)
        self.params = params
        self.railway_encoding = None
        self.normalize = normalize
        self.state_size = self._get_state_size()
        self.deadlocks_detector = DeadlocksDetector()
        self.partial_rewards = dict()
        self.arrived_turns = dict()
        self.stop_actions = dict()

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
        return num_deadlocks == len(remaining_agents) and len(remaining_agents) > 0

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
        for i, agent in enumerate(self.agents):
            if activate_agents:
                self.set_agent_active(agent)
            self._break_agent(agent)
            if agent.malfunction_data["malfunction"] > 0:
                agent.speed_data['transition_action_on_cellexit'] = RailEnvActions.DO_NOTHING
            self._fix_agent_after_malfunction(agent)

            # Reset partial rewards
            self.partial_rewards[i] = 0.0

        # Reset common variables
        self.num_resets += 1
        self._elapsed_steps = 0
        self.dones = dict.fromkeys(
            list(range(self.get_num_agents())) + ["__all__"], False
        )
        self.arrived_turns = [None] * self.get_num_agents()
        self.stop_actions = [0] * self.get_num_agents()

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
            'status': {}, 'deadlocks': {}, 'deadlock_turns': {}
        }
        for i, agent in enumerate(self.agents):
            info_dict['action_required'][i] = self.action_required(agent)
            info_dict['malfunction'][i] = agent.malfunction_data['malfunction']
            info_dict['speed'][i] = agent.speed_data['speed']
            info_dict['status'][i] = agent.status
            info_dict["deadlocks"][i] = self.deadlocks_detector.deadlocks[i]
            info_dict["deadlock_turns"][i] = self.deadlocks_detector.deadlock_turns[i]

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

    def step(self, action_dict_):
        '''
        Perform a step in the environment
        '''
        agents_in_decision_cells = self.agents_in_decision_cells()
        obs, rewards, dones, info = super().step(action_dict_)
        info["deadlocks"], info["deadlock_turns"] = self.deadlocks_detector.step(
            self
        )

        # Patch dones dict, update arrived agents turns and stop actions
        remove_all = False
        for agent in range(self.get_num_agents()):
            if dones[agent] and not self.railway_encoding.is_done(agent):
                dones[agent] = False
                remove_all = True
            if dones[agent] and self.arrived_turns[agent] is None:
                self.arrived_turns[agent] = self._elapsed_steps - 1

            # Store the number of consequent stop actions
            if action_dict_[agent] == RailEnvActions.STOP_MOVING.value:
                self.stop_actions[agent] += 1
            else:
                self.stop_actions[agent] = 0

        # If at least one agent is not at target, then
        # the __all__ flag of dones should be False
        if remove_all:
            dones["__all__"] = False

        # Compute custom rewards
        custom_rewards = self._reward_shaping(
            action_dict_, rewards, dones, info, agents_in_decision_cells
        )

        return (self._normalize_obs(obs), rewards, custom_rewards, dones, info)

    def _reward_shaping(self, action_dict_, rewards, dones, info, agents_in_decision_cells):
        '''
        Apply custom reward functions
        '''
        # The step for which we are evaluating the rewards
        # is the previous one
        step = self._elapsed_steps - 1

        custom_rewards = copy.deepcopy(rewards)
        for agent in range(self.get_num_agents()):
            # Return a positive reward equal to the maximum number of steps
            # minus the current number of steps if the agent has arrived
            if dones[agent] and self.arrived_turns[agent] == step:
                done_reward = self._max_episode_steps - step
                custom_rewards[agent] = done_reward
                self.partial_rewards[agent] = done_reward
            # Return minus the maximum number of steps if the agent is in deadlock
            elif info["deadlocks"][agent] and info["deadlock_turns"][agent] == step:
                deadlock_penalty = -self._max_episode_steps
                custom_rewards[agent] = deadlock_penalty
                self.partial_rewards[agent] = deadlock_penalty
            # Accumulate rewards for choices if the agent is not in a decision cell
            # and add other penalties, such as the stop moving one
            else:
                self.partial_rewards[agent] += custom_rewards[agent]
                # If an agent performed a STOP action, give the agent a reward
                # which is worse than the the reward it could have received
                # by choosing any other action
                if action_dict_[agent] == RailEnvActions.STOP_MOVING.value:
                    weight = self.railway_encoding.stop_moving_worst_alternative_weight(
                        agent
                    )
                    weight *= (1 / self.agents[agent].speed_data['speed'])
                    weight *= (
                        self.stop_actions[agent] *
                        self.params.env.rewards.stop_penalty
                    )
                    self.partial_rewards[agent] += -(weight + 1)
                custom_rewards[agent] = 0.0

            # Reset the partial rewards counter when an agent is
            # in a decision cell or in the last step
            if agents_in_decision_cells[agent] or self._elapsed_steps == self._max_episode_steps:
                custom_rewards[agent] = self.partial_rewards[agent]
                self.partial_rewards[agent] = 0.0

            # Normalize rewards
            custom_rewards[agent] /= self._max_episode_steps

        return custom_rewards

    def agents_in_decision_cells(self):
        '''
        Return the agents that are on a decision cell
        '''
        return [
            self.railway_encoding.is_real_decision(h)
            for h in range(self.get_num_agents())
        ]

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
                        obs[handle], self.obs_builder.max_depth,
                        self.params.observator.tree.radius
                    )

        return obs
