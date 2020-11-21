import os
from enum import IntEnum

from flatland.envs.agent_utils import RailAgentStatus
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.rail_env import RailEnvActions
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator

import utils


class RailEnvChoices(IntEnum):

    CHOICE_LEFT = 0
    CHOICE_RIGHT = 1
    STOP = 2

    @staticmethod
    def value_of(value):
        '''
        Return an instance of RailEnvChoices from the given choice type int
        '''
        for _, choice_type in RailEnvChoices.__members__.items():
            if choice_type.value == value.capitalize():
                return choice_type
        return None

    @staticmethod
    def values():
        '''
        Return a list of every possible RailEnvChoices
        '''
        return [
            choice_type
            for _, choice_type in RailEnvChoices.__members__.items()
        ]


def agent_action(original_dir, final_dir):
    '''
    Return the action performed by an agent, by analyzing
    the starting direction and the final direction of the movement
    '''
    value = (final_dir.value - original_dir.value) % 4
    if value in (1, -3):
        return RailEnvActions.MOVE_RIGHT
    elif value in (-1, 3):
        return RailEnvActions.MOVE_LEFT
    return RailEnvActions.MOVE_FORWARD


def create_save_env(path, width, height, num_trains, max_cities, max_rails_between_cities, max_rails_in_cities, grid=False, seed=0):
    '''
    Create a RailEnv environment with the given settings and save it as pickle
    '''
    rail_generator = sparse_rail_generator(
        max_num_cities=max_cities,
        seed=seed,
        grid_mode=grid,
        max_rails_between_cities=max_rails_between_cities,
        max_rails_in_city=max_rails_in_cities,
    )
    env = RailEnv(
        width=width,
        height=height,
        rail_generator=rail_generator,
        number_of_agents=num_trains
    )
    save_env(path, env)


def save_env(path, env):
    '''
    Save the given RailEnv environment as pickle
    '''
    filename = os.path.join(
        path,
        f"{env.width}x{env.height}-{env.random_seed}.pkl"
    )
    RailEnvPersister.save(env, filename)


def get_seed(env, seed=None):
    '''
    Exploit the RailEnv to get a random seed
    '''
    seed = env._seed(seed)
    return seed[0]


def check_if_all_blocked(env):
    '''
    Checks whether all the agents are blocked (full deadlock situation)
    '''
    # First build a map of agents in each position
    location_has_agent = {}
    for agent in env.agents:
        if agent.status in [RailAgentStatus.ACTIVE, RailAgentStatus.DONE] and agent.position:
            location_has_agent[tuple(agent.position)] = 1

    # Looks for any agent that can still move
    for handle in env.get_agent_handles():
        agent = env.agents[handle]
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        elif agent.status == RailAgentStatus.DONE:
            agent_virtual_position = agent.target
        else:
            continue

        possible_transitions = env.rail.get_transitions(
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


def get_agents_same_start(env):
    '''
    Return a dictionary indexed by agents starting positions,
    and having a list of handles as values, s.t. agents with
    the same starting position are ordered by decreasing speed
    '''
    agents_with_same_start = dict()
    for handle_one, agent_one in enumerate(env.agents):
        for handle_two, agent_two in enumerate(env.agents):
            if handle_one != handle_two and agent_one.initial_position == agent_two.initial_position:
                agents_with_same_start.setdefault(
                    agent_one.initial_position, set()
                ).update({handle_one, handle_two})

    for position in agents_with_same_start:
        agents_with_same_start[position] = sorted(
            list(agents_with_same_start[position]), reverse=True,
            key=lambda x: env.agents[x].speed_data['speed']
        )
    return agents_with_same_start
