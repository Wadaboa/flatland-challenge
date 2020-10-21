
from timeit import default_timer
import numpy as np


def find_duplicates(arr):
    '''
    Return a list, containing indexes of duplicate elements
    '''
    dups = []
    for i, elem in enumerate(arr):
        if elem is not None:
            dups.extend(find_duplicate(arr, elem, i, starting=i + 1))
    return dups


def find_duplicate(arr, elem, index, starting=0):
    dups = []
    for i, elem_one in enumerate(arr[starting:]):
        if elem_one is not None and i != index and elem_one == elem:
            dups.append((elem_one, {index, i}))
    return dups


def fill_none(mat, lenght):
    '''
    Fill a matrix to have the same number of elements in each row
    '''
    for i, row in enumerate(mat):
        if len(row) < lenght:
            mat[i].extend([None] * (lenght - len(row)))
    return mat


def get_index(arr, elem):
    return arr.index(elem) if elem in arr else None


class Timer(object):
    """
    Utility to measure times.

    TODO:
    - add "lap" method to make it easier to measure average time (+std) when measuring the same thing multiple times.
    """

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
