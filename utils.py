
import numpy as np


def duplicates(arr):
    '''
    Return a list, containing indexes of duplicate elements
    '''
    dups = []
    for i, elem_one in enumerate(arr):
        dup = set()
        for j, elem_two in enumerate(arr[i + 1:]):
            if elem_one == elem_two and elem_one != np.nan:
                dup.add(i)
                dup.add(i + 1 + j)
        if dup:
            dups.append([elem_one, dup])
    return dups


def fill_none(mat, lenght):
    '''
    Fill a matrix to have the same number of elements in each row
    '''
    for row in mat:
        if len(row) < lenght:
            row.extend([None] * (len(row) - lenght))
    return mat
