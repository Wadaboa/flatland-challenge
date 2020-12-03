import os
import copy
from enum import IntEnum

from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import RailEnvActions
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import rail_from_file, sparse_rail_generator
from flatland.envs.malfunction_generators import ParamMalfunctionGen, MalfunctionParameters
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

from predictions import ShortestPathPredictor, NullPredictor
from binary_tree_obs import BinaryTreeObservator
from graph_obs import GraphObservator
from environments import RailEnvWrapper

import utils


OBSERVATORS = {
    "tree": TreeObsForRailEnv,
    "binary_tree": BinaryTreeObservator,
    "graph": GraphObservator
}
PREDICTORS = {
    "tree": ShortestPathPredictorForRailEnv,
    "binary_tree": ShortestPathPredictor,
    "graph": ShortestPathPredictor
}


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

    @staticmethod
    def choice_size():
        '''
        Return the number of values that can be assigned 
        to a RailEnvChoices instance
        '''
        return len(RailEnvChoices.values())

    @staticmethod
    def default_choices():
        '''
        Return a mask of choices, s.t. the only choice that
        can always be applied is STOP
        '''
        return [False, False, True]


def create_rail_env(args, load_env=""):
    '''
    Build a RailEnv object with the specified parameters,
    as described in the .yml file
    '''
    # Check if an environment file is provided
    if load_env:
        rail_generator = rail_from_file(load_env)
    else:
        rail_generator = sparse_rail_generator(
            max_num_cities=args.env.max_cities,
            grid_mode=args.env.grid,
            max_rails_between_cities=args.env.max_rails_between_cities,
            max_rails_in_city=args.env.max_rails_in_cities,
            seed=args.env.seed
        )

    # Build predictor and observator
    obs_type = args.policy.type.get_true_key()
    predictor = PREDICTORS[obs_type](max_depth=args.predictor.max_depth)
    observator = OBSERVATORS[obs_type](args.observator.max_depth, predictor)

    # Initialize malfunctions
    malfunctions = None
    if args.env.malfunctions.enabled:
        malfunctions = ParamMalfunctionGen(
            MalfunctionParameters(
                malfunction_rate=args.env.malfunctions.rate,
                min_duration=args.env.malfunctions.min_duration,
                max_duration=args.env.malfunctions.max_duration
            )
        )

    # Initialize agents speeds
    speed_map = None
    if args.env.variable_speed:
        speed_map = {
            1.: 0.25,
            1. / 2.: 0.25,
            1. / 3.: 0.25,
            1. / 4.: 0.25
        }
    schedule_generator = sparse_schedule_generator(
        speed_map, seed=args.env.seed
    )

    # Build the environment
    return RailEnvWrapper(
        params=args,
        width=args.env.width,
        height=args.env.height,
        rail_generator=rail_generator,
        schedule_generator=schedule_generator,
        number_of_agents=args.env.num_trains,
        obs_builder_object=observator,
        malfunction_generator=malfunctions,
        remove_agents_at_target=True,
        random_seed=args.env.seed
    )


def create_save_env(path, width, height, num_trains, max_cities,
                    max_rails_between_cities, max_rails_in_cities, grid=False, seed=0):
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
    env.save(path)


def get_seed(env, seed=None):
    '''
    Exploit the RailEnv to get a random seed
    '''
    seed = env._seed(seed)
    return seed[0]


def copy_obs(obs):
    '''
    Return a deep copy of the given observation
    '''
    if hasattr(obs, "copy"):
        return obs.copy()
    return copy.deepcopy(obs)


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
