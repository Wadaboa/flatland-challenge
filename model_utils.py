import numpy as np
import numpy.ma as ma


def masked_softmax(vec, mask, dim=1, fill_value=0, temperature=1):
    '''
    Softmax only on valid outputs
    '''
    exps = np.exp(vec / temperature)
    masked_arr = ma.masked_array(
        exps, mask=np.invert(mask), fill_value=fill_value
    )
    res = masked_arr / masked_arr.sum(axis=dim, keepdims=True)
    res[res.mask] = fill_value
    return res.data


def masked_max(vec, mask, dim=1, fill_value=0):
    '''
    Max only on valid outputs
    '''
    masked_max_arr = ma.masked_array(
        vec, mask=np.invert(mask), fill_value=fill_value
    ).max(axis=dim, keepdims=True, fill_value=fill_value)
    masked_max_arr[masked_max_arr.mask] = fill_value
    return masked_max_arr.data


def masked_argmax(vec, mask, dim=1, fill_value=np.nan):
    '''
    Argmax only on valid outputs
    '''
    masked_argmax_arr = ma.masked_array(
        vec, mask=np.invert(mask), fill_value=fill_value
    ).argmax(axis=dim, fill_value=fill_value)

    # Argmax on masked arrays returns a plain numpy array,
    # so we have to reshape it
    if dim > 0:
        new_shape = list(vec.shape)
        new_shape[dim] = 1
        masked_argmax_arr = masked_argmax_arr.reshape(tuple(new_shape))

    return masked_argmax_arr
