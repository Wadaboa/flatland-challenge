import random
import os
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
    return abs(a - b) <= rtol


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
        else:
            values[:] = under
    except:
        pass
    values[values == -np.inf] = under
    values[values == np.inf] = over
    return values


def extract_fov(matrix, center_index, window_size, pad=0):
    '''
    Extract a patch of size window_size from the given matrix centered around
    the specified position and pad external values with the given fill value
    '''
    # Window is entirely contained in the given matrix
    m, n = matrix.shape
    offset = window_size // 2
    yl, yu = center_index[0] - offset, center_index[0] + offset
    xl, xu = center_index[1] - offset, center_index[1] + offset
    if xl >= 0 and xu < n and yl >= 0 and yu < m:
        return np.array(matrix[yl: yu + 1, xl:xu + 1], dtype=matrix.dtype)

    # Window has to be padded
    window = np.full((window_size, window_size), pad, dtype=matrix.dtype)
    c_yl, c_yu = np.clip(yl, 0, m), np.clip(yu, 0, m)
    c_xl, c_xu = np.clip(xl, 0, n), np.clip(xu, 0, n)
    sub = matrix[c_yl: c_yu + 1, c_xl:c_xu + 1]
    w_yl = 0 if yl >= 0 else abs(yl)
    w_yu = window_size if yu < m else window_size - (yu - m) - 1
    w_xl = 0 if xl >= 0 else abs(xl)
    w_xu = window_size if xu < n else window_size - (xu - n) - 1
    window[w_yl:w_yu, w_xl:w_xu] = sub
    return window


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


def set_num_threads(num_threads):
    '''
    Set the maximum number of threads PyTorch can use
    '''
    torch.set_num_threads(num_threads)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)


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


class Struct:
    '''
    Struct class, s.t. a nested dictionary is transformed
    into a nested object
    '''

    def __init__(self, **entries):
        for k, v in entries.items():
            if isinstance(v, dict):
                self.__dict__.update({k: Struct(**v)})
            else:
                self.__dict__.update({k: v})

    def get_true_key(self):
        '''
        Return the only key in the Struct s.t. its value is True
        '''
        true_types = [k for k, v in self.__dict__.items() if v == True]
        assert len(true_types) == 1
        return true_types[0]

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)
