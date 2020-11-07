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
