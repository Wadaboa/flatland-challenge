import os
import copy

import numpy as np

from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.persistence import RailEnvPersister
from flatland.utils.rendertools import RenderTool, AgentRenderVariant

from obs import normalization
from env import env_utils
from env.deadlocks import DeadlocksDetector
from env.railway_encoding import CellOrientationGraph
from obs.binary_tree import BinaryTreeObservator
from obs.single_agent_graph import SingleAgentGraphObservator
from obs.fov import FOVObservator


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
        self.current_info = dict()
        self.num_actions = self.get_num_actions()

    def _get_state_size(self):
        '''
        Compute the state size based on observation type
        '''
        n_features_per_node = self.obs_builder.observation_dim
        n_nodes = 1
        if isinstance(self.obs_builder, TreeObsForRailEnv):
            n_nodes = sum(
                4 ** i for i in range(self.obs_builder.max_depth + 1)
            )
        elif isinstance(self.obs_builder, BinaryTreeObservator):
            n_nodes = sum(2 ** i for i in range(self.obs_builder.max_depth))
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
        self.current_info = {
            'action_required': {}, 'malfunction': {}, 'speed': {},
            'status': {}, 'deadlocks': {}, 'deadlock_turns': {}, 'finished': {},
            'first_time_deadlock': {}, 'first_time_finished': {}
        }
        for i, agent in enumerate(self.agents):
            self.current_info['action_required'][i] = self.action_required(
                agent
            )
            self.current_info['malfunction'][i] = agent.malfunction_data['malfunction']
            self.current_info['speed'][i] = agent.speed_data['speed']
            self.current_info['status'][i] = agent.status
            self.current_info["deadlocks"][i] = self.deadlocks_detector.deadlocks[i]
            self.current_info["deadlock_turns"][i] = self.deadlocks_detector.deadlock_turns[i]
            self.current_info["finished"][i] = self.dones[i] or self.deadlocks_detector.deadlocks[i]
            self.current_info["first_time_deadlock"][i] = (
                self.deadlocks_detector.deadlocks[i] and
                0 == self.deadlocks_detector.deadlock_turns[i]
            )
            self.current_info["first_time_finished"][i] = (
                self.dones[i] and
                0 == self.arrived_turns[i]
            )

        # Return the new observation vectors for each agent
        observation_dict = self._get_observations()
        return (self._normalize_obs(observation_dict), self.current_info)

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
        current_step = self._elapsed_steps
        agents_in_decision_cells = self.agents_in_decision_cells()
        obs, rewards, dones, info = super().step(action_dict_)
        info["deadlocks"], info["deadlock_turns"] = self.deadlocks_detector.step(
            self
        )
        finished = dict()
        first_time_deadlock = dict()
        first_time_finished = dict()
        # Patch dones dict, update arrived agents turns and stop actions
        remove_all = False
        for agent in range(self.get_num_agents()):
            if dones[agent] and not self.railway_encoding.is_done(agent):
                dones[agent] = False
                remove_all = True
            if dones[agent] and self.arrived_turns[agent] is None:
                self.arrived_turns[agent] = current_step
                first_time_finished[agent] = True
            else:
                first_time_finished[agent] = False

            # Store the number of consequent stop actions
            if action_dict_[agent] == RailEnvActions.STOP_MOVING.value:
                self.stop_actions[agent] += 1
            else:
                self.stop_actions[agent] = 0
            finished[agent] = dones[agent] or info['deadlocks'][agent]
            first_time_deadlock[agent] = (
                info['deadlocks'][agent] and
                current_step == info['deadlock_turns'][agent]
            )

        info['finished'] = finished
        info['first_time_finished'] = first_time_finished
        info['first_time_deadlock'] = first_time_deadlock

        # If at least one agent is not at target, then
        # the __all__ flag of dones should be False
        if remove_all:
            dones["__all__"] = False

        # Compute custom rewards
        custom_rewards = self._reward_shaping(
            action_dict_, rewards, dones, info, agents_in_decision_cells
        )

        self.current_info = info

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
                    # Clip stop reward to be at maximum equal to the deadlock penalty
                    self.partial_rewards[agent] += np.clip(
                        -weight, -self._max_episode_steps, 0
                    )
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

    def agents_adjacency_matrix(self, radius=None):
        '''
        Return the adjacency matrix containing pairwise distances between agents
        '''
        adj = np.zeros((self.get_num_agents(), self.get_num_agents()))
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if i != j:
                    distance = self.railway_encoding.get_agents_distance(i, j)
                    if (distance is not None and
                            (radius is None or (radius is not None and distance <= radius))):
                        adj[i, j] = distance
        return adj

    def pre_act(self):
        '''
        Return the list of legal actions and choices for each agent and a list
        representing which agent needs to make a choice
        '''
        legal_choices = np.full(
            (self.get_num_agents(), env_utils.RailEnvChoices.choice_size()),
            env_utils.RailEnvChoices.default_choices()
        )
        legal_actions = np.full(
            (self.get_num_agents(), self.num_actions), False
        )
        update_values = np.full((self.get_num_agents(),), False)
        for agent in self.get_agent_handles():
            legal_actions[
                agent, self.railway_encoding.get_agent_actions(agent)
            ] = True
            if self.current_info['action_required'][agent] and self.railway_encoding.is_real_decision(agent):
                legal_choices[agent] = self.railway_encoding.get_legal_choices(
                    agent, legal_actions[agent]
                )
                update_values[agent] = True
        print(legal_actions.shape)
        return legal_actions, legal_choices, update_values

    def post_act(self, choices, is_best, legal_actions, update_values):
        '''
        '''
        action_dict = dict()
        choices_count = np.zeros((env_utils.RailEnvChoices.choice_size(),))
        choices_taken = choices[~update_values]
        num_exploration_choices = np.zeros_like(choices_count)
        choice_dict = dict()

        for agent in self.get_agent_handles():
            if update_values[agent]:
                action = self.railway_encoding.map_choice_to_action(
                    choices[agent], legal_actions[agent]
                )
                if choices[agent] == 1:
                    print(action, choices[agent], legal_actions[agent])
                assert action != RailEnvActions.DO_NOTHING.value, (
                    choices[agent], legal_actions[agent]
                )
                choices_count[choices[agent]] += 1
                num_exploration_choices[choices[agent]] += int(
                    not(is_best[agent])
                )
                choice_dict.update({agent: choices[agent]})
            elif not self.dones[agent] and self.agents[agent].speed_data['position_fraction'] == 0:
                actions = np.flatnonzero(legal_actions[agent])
                assert actions.shape[0] > 0, actions
                if actions.shape[0] > 1:
                    action = RailEnvActions.DO_NOTHING.value
                else:
                    action = actions[0]
            else:
                action = RailEnvActions.DO_NOTHING.value
            action_dict.update({agent: action})
        metadata = {
            'choices_count': choices_count,
            'choices_taken': choices_taken,
            'num_exploration_choices': num_exploration_choices,
            'choice_dict': choice_dict
        }
        print(choice_dict)

        return action_dict, metadata

    def pre_step(self, experience):
        (prev_obs, prev_choices, custom_rewards,
         obs, legal_choices, update_values) = experience
        finished = np.array(list(self.current_info['finished'].values()))
        experiences = []
        for agent in self.get_agent_handles():
            if (update_values[agent] or
                    self.current_info['first_time_finished'][agent] or
                    self.current_info['first_time_deadlock'][agent]):
                if self.params.policy.type.multi_agent_graph:
                    exp = (
                        list(prev_obs[agent].values()),
                        list(prev_choices[agent].values()),
                        np.array(list(custom_rewards.values())),
                        list(obs.values()),
                        np.array(list(legal_choices.values())),
                        finished, update_values
                    )
                else:
                    exp = (
                        prev_obs[agent], prev_choices[agent], custom_rewards[agent],
                        obs[agent], legal_choices[agent], finished[agent], update_values[agent]
                    )
                experiences.append(exp)
        return experiences

    def post_step(self, obs, choice_dict, next_obs, update_values, rewards, custom_rewards):
        prev_obs = dict()
        prev_choices = dict()
        score = 0.0
        custom_score = 0.0
        for agent in self.get_agent_handles():
            if update_values[agent] or self.current_info['first_time_finished'][agent] or self.current_info['first_time_deadlock'][agent]:
                if self.params.policy.type.multi_agent_graph:
                    prev_obs[agent] = env_utils.copy_obs(obs)
                    prev_choices[agent] = choice_dict
                else:
                    prev_obs[agent] = env_utils.copy_obs(obs[agent])
                    prev_choices[agent] = choice_dict[agent]

            # Update observation and score
            score += rewards[agent]
            custom_score += custom_rewards[agent]
            if next_obs[agent] is not None:
                obs[agent] = env_utils.copy_obs(next_obs[agent])
        metadata = {
            'obs': obs,
            'prev_obs': prev_obs,
            'prev_choices': prev_choices,
            'score': score,
            'custom_score': custom_score
        }
        return metadata

    def get_num_actions(self):
        '''
        Return the number of possible RailEnvActions
        '''
        return len([
            action_type for _, action_type in RailEnvActions.__members__.items()
        ])

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
                    obs[handle] = normalization.normalize_tree_obs(
                        obs[handle], self.obs_builder.max_depth,
                        self.params.observator.tree.radius
                    )

        return obs
