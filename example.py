import numpy as np
import matplotlib.pyplot as plt
import time
import itertools

from flatland.envs.malfunction_generators import malfunction_from_params
from flatland.envs.observations import GlobalObsForRailEnv, TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env import RailEnvActions
from flatland.envs.rail_generators import sparse_rail_generator, complex_rail_generator, random_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool, AgentRenderVariant

import networkx as nx
from railway_encoding import CellOrientationGraph
from observations import CustomObservation
from predictions import ShortestPathPredictor

width = 16
height = 16
nr_trains = 1
cities_in_map = 2
seed = 14
# Type of city distribution, if False cities are randomly placed
grid_distribution_of_cities = False
# Max number of tracks allowed between cities. This is number of entry point to a city
max_rails_between_cities = 3
# Max number of parallel tracks within a city, representing a realistic trainstation
max_rails_in_city = 1


rail_generator = sparse_rail_generator(max_num_cities=cities_in_map,
                                       seed=seed,
                                       grid_mode=grid_distribution_of_cities,
                                       max_rails_between_cities=max_rails_between_cities,
                                       max_rails_in_city=max_rails_in_city,
                                       )
'''
rail_generator = random_rail_generator(
    cell_type_relative_proportion=[1.0] * 11, seed=1)
'''
# Custom observation builder without predictor
# observation_builder = GlobalObsForRailEnv()

# Custom observation builder with predictor, uncomment line below if you want to try this one
observation_builder = CustomObservation(
    predictor=ShortestPathPredictor(max_depth=20)
)

# Construct the enviornment with the given observation, generataors, predictors, and stochastic data
env = RailEnv(width=width,
              height=height,
              rail_generator=rail_generator,
              number_of_agents=nr_trains,
              obs_builder_object=observation_builder,
              remove_agents_at_target=True)
env.reset()


# Initiate the renderer
env_renderer = RenderTool(env, gl="PILSVG",
                          agent_render_variant=AgentRenderVariant.ONE_STEP_BEHIND,
                          show_debug=True,
                          screen_height=1000,
                          screen_width=1000)

env_renderer.render_env(show=True)
time.sleep(10)

# Import your own Agent or use RLlib to train agents on Flatland
# As an example we use a random agent instead


class RandomAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

    def act(self, state):
        """
        :param state: input is the observation of the agent
        :return: returns an action
        """
        return np.random.choice([RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_RIGHT, RailEnvActions.MOVE_LEFT,
                                 RailEnvActions.STOP_MOVING])

    def step(self, memories):
        """
        Step function to improve agent by adjusting policy given the observations

        :param memories: SARS Tuple to be
        :return:
        """
        return

    def save(self, filename):
        # Store the current policy
        return

    def load(self, filename):
        # Load a policy
        return


# Initialize the agent with the parameters corresponding to the environment and observation_builder
controller = RandomAgent(218, env.action_space[0])

# We start by looking at the information of each agent
# We can see the task assigned to the agent by looking at
print("\n Agents in the environment have to solve the following tasks: \n")
for agent_idx, agent in enumerate(env.agents):
    print(
        "The agent with index {} has the task to go from its initial position {}, facing in the direction {} to its target at {}.".format(
            agent_idx, agent.initial_position, agent.direction, agent.target))

# The agent will always have a status indicating if it is currently present in the environment or done or active
# For example we see that agent with index 0 is currently not active
print("\n Their current statuses are:")
print("============================")

for agent_idx, agent in enumerate(env.agents):
    print("Agent {} status is: {} with its current position being {}".format(agent_idx, str(agent.status),
                                                                             str(agent.position)))

# The agent needs to take any action [1,2,3] except do_nothing or stop to enter the level
# If the starting cell is free they will enter the level
# If multiple agents want to enter the same cell at the same time the lower index agent will enter first.

# Let's check if there are any agents with the same start location
agents_with_same_start = set()
print("\n The following agents have the same initial position:")
print("=====================================================")
for agent_idx, agent in enumerate(env.agents):
    for agent_2_idx, agent2 in enumerate(env.agents):
        if agent_idx != agent_2_idx and agent.initial_position == agent2.initial_position:
            print("Agent {} as the same initial position as agent {}".format(
                agent_idx, agent_2_idx))
            agents_with_same_start.add(agent_idx)

# Lets try to enter with all of these agents at the same time
action_dict = dict()

for agent_id in agents_with_same_start:
    action_dict[agent_id] = 1  # Try to move with the agents

# Do a step in the environment to see what agents entered:
env.step(action_dict)

# Current state and position of the agents after all agents with same start position tried to move
print("\n This happened when all tried to enter at the same time:")
print("========================================================")
for agent_id in agents_with_same_start:
    print(
        "Agent {} status is: {} with the current position being {}.".format(
            agent_id, str(env.agents[agent_id].status),
            str(env.agents[agent_id].position)))

# As you see only the agents with lower indexes moved. As soon as the cell is free again the agents can attempt
# to start again.

# You will also notice, that the agents move at different speeds once they are on the rail.
# The agents will always move at full speed when moving, never a speed inbetween.
# The fastest an agent can go is 1, meaning that it moves to the next cell at every time step
# All slower speeds indicate the fraction of a cell that is moved at each time step
# Lets look at the current speed data of the agents:

print("\n The speed information of the agents are:")
print("=========================================")

for agent_idx, agent in enumerate(env.agents):
    print(
        "Agent {} speed is: {:.2f} with the current fractional position being {}".format(
            agent_idx, agent.speed_data['speed'], agent.speed_data['position_fraction']))

# New the agents can also have stochastic malfunctions happening which will lead to them being unable to move
# for a certain amount of time steps. The malfunction data of the agents can easily be accessed as follows
print("\n The malfunction data of the agents are:")
print("========================================")

for agent_idx, agent in enumerate(env.agents):
    print(
        "Agent {} is OK = {}".format(
            agent_idx, agent.malfunction_data['malfunction'] < 1))

# Now that you have seen these novel concepts that were introduced you will realize that agents don't need to take
# an action at every time step as it will only change the outcome when actions are chosen at cell entry.
# Therefore the environment provides information about what agents need to provide an action in the next step.
# You can access this in the following way.

# Chose an action for each agent
for a in range(env.get_num_agents()):
    action = controller.act(0)
    action_dict.update({a: action})
# Do the environment step
observations, rewards, dones, information = env.step(action_dict)
print("\n The following agents can register an action:")
print("========================================")
for info in information['action_required']:
    print("Agent {} needs to submit an action.".format(info))

# We recommend that you monitor the malfunction data and the action required in order to optimize your training
# and controlling code.

# Let us now look at an episode playing out with random actions performed

print("\nStart episode...")

# Reset the rendering system
env_renderer.reset()

# Here you can also further enhance the provided observation by means of normalization
# See training navigation example in the baseline repository


score = 0
# Run episode
frame_step = 0

for step in range(500):
    # Chose an action for each agent in the environment
    for a in range(env.get_num_agents()):
        action = controller.act(observations[a])
        action_dict.update({a: action})

    # Environment step which returns the observations for all agents, their corresponding
    # reward and whether their are done

    next_obs, all_rewards, done, _ = env.step(action_dict)

    env_renderer.render_env(
        show=True, show_observations=True, show_predictions=True)

    frame_step += 1
    # Update replay buffer and train agent
    for a in range(env.get_num_agents()):
        controller.step(
            (observations[a], action_dict[a], all_rewards[a], next_obs[a], done[a]))
        score += all_rewards[a]

    observations = next_obs.copy()
    # print(observations)
    if done['__all__']:
        break
    print('Episode: Steps {}\t Score = {}'.format(step, score))
    time.sleep(5)
