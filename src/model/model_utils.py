import numpy as np
import torch.nn as nn


def masked_softmax(vec, mask, dim=1, temperature=1):
    '''
    Softmax only on valid outputs
    '''
    assert vec.shape == mask.shape
    assert np.all(mask.astype(bool).any(axis=dim)), mask

    exps = vec.copy()
    exps = np.exp(vec / temperature)
    exps[~mask.astype(bool)] = 0
    return exps / exps.sum(axis=dim, keepdims=True)


def masked_max(vec, mask, dim=1):
    '''
    Max only on valid outputs
    '''
    assert vec.shape == mask.shape
    assert np.all(mask.astype(bool).any(axis=dim)), mask

    res = vec.copy()
    res[~mask.astype(bool)] = np.nan
    return np.nanmax(res, axis=dim, keepdims=True)


def masked_argmax(vec, mask, dim=1):
    '''
    Argmax only on valid outputs
    '''
    assert vec.shape == mask.shape
    assert np.all(mask.astype(bool).any(axis=dim)), mask

    res = vec.copy()
    res[~mask.astype(bool)] = np.nan
    argmax_arr = np.nanargmax(res, axis=dim)

    # Argmax has no keepdims argument
    if dim > 0:
        new_shape = list(res.shape)
        new_shape[dim] = 1
        argmax_arr = argmax_arr.reshape(tuple(new_shape))

    return argmax_arr


def conv_block_output_size(modules, input_width, input_height):
    '''
    Given a sequence of PyTorch modules (e.g. Python list, PyTorch Sequential/ModuleList)
    containing convolution related layers (currently only Conv2d and MaxPool2d are supported),
    returns the output size of the input tensor, after it passes through all the given layers
    '''
    output_width, output_height = input_width, input_height
    for module in modules:
        kernel_size = module.kernel_size
        stride = module.stride
        padding = module.padding
        dilation = module.dilation
        if type(module) in (nn.Conv2d, nn.MaxPool2d):
            output_width = np.floor((
                output_width + 2 * padding - dilation * (kernel_size - 1) - 1
            ) / stride + 1)
            output_height = np.floor((
                output_height + 2 * padding - dilation * (kernel_size-1) - 1
            ) / stride + 1)
    return output_width, output_height
