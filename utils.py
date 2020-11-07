import random
import os
from timeit import default_timer

import numpy as np
import torch

from flatland.envs.persistence import RailEnvPersister


def get_index(arr, elem):
    return arr.index(elem) if elem in arr else None


def reciprocal_sum(a, b):
    return (1 / a) + (1 / b)


def min_max_scaling(values, lower, upper, under, over, known_min=None, known_max=None):
    '''
    Perform min-max scaling over the given array
    (`under` is substituted for -np.inf and `over` for np.inf)
    '''
    finite_values = values[np.isfinite(values)]
    min_value, max_value = known_min, known_max
    try:
        if min_value is None:
            min_value = finite_values.min()
        if max_value is None:
            max_value = finite_values.max()
        if min_value != max_value:
            values = lower + (
                ((values - min_value) * (upper - lower)) /
                (max_value - min_value)
            )
        elif min_value != 0:
            values = values / min_value
    except:
        pass
    values[values == -np.inf] = under
    values[values == np.inf] = over
    return values


def fix_random(seed):
    '''
    Fix all the possible sources of randomness
    '''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_save_env(path, width, height, num_trains, max_cities, max_rails_between_cities, max_rails_in_cities, grid=False, seed=0):
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
    filename = os.path.join(
        path,
        f"{env.width}x{env.height}-{env.random_seed}.pkl"
    )
    RailEnvPersister.save(env, filename)


def get_seed(env, seed=None):
    return env._seed(seed)[0]


class Timer():
    '''
    Utility to measure times
    '''

    def __init__(self):
        self.total_time = 0.0
        self.start_time = 0.0
        self.end_time = 0.0

    def start(self):
        self.start_time = default_timer()

    def end(self):
        self.total_time += default_timer() - self.start_time

    def get(self):
        return self.total_time

    def get_current(self):
        return default_timer() - self.start_time

    def reset(self):
        self.__init__()

    def __repr__(self):
        return self.get()
