import random
from timeit import default_timer

import numpy as np
import torch


def get_index(arr, elem):
    '''
    Return the index of the first occurrence of `elem` in `arr`,
    if `elem` is present in `arr`, otherwise return None
    '''
    return arr.index(elem) if elem in arr else None


def is_close(a, b, rtol=1e-03):
    '''
    Return if a is relatively close to the value of b
    '''
    return abs(a-b) <= rtol


def reciprocal_sum(a, b):
    '''
    Compute the reciprocal sum of the given inputs
    '''
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
