
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
