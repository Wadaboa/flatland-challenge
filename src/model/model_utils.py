import numpy as np
import torch
import torch.nn as nn


def get_linear(input_size, output_size, hidden_sizes, nonlinearity="tanh"):
    '''
    Returns a PyTorch Sequential object containing FC layers with
    non-linear activation functions, by following the given input/hidden/output sizes
    '''
    fc = []
    nl = nn.ReLU(inplace=True) if nonlinearity == "relu" else nn.Tanh()
    sizes = [input_size] + hidden_sizes + [output_size]
    for i in range(1, len(sizes)):
        fc.extend([nn.Linear(sizes[i - 1], sizes[i]), nl])
    return nn.Sequential(*fc)


def conv_bn_act(input_channels, output_channels, kernel_size=3,
                stride=1, padding=0, nonlinearity="relu"):
    '''
    Returns a block composed by a convolutional layer and a batch norm one,
    followed by a non-linearity (e.g. ReLU or Tanh)
    '''
    return [
        nn.Conv2d(
            input_channels, output_channels,
            kernel_size=kernel_size, stride=stride, padding=padding
        ),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(inplace=True) if nonlinearity == "relu" else nn.Tanh()
    ]


def get_conv(input_channels, output_channels, hidden_channels,
             conv_params, pool_params, nonlinearity="relu"):
    '''
    Returns a PyTorch Sequential object containing `conv_bn_act` blocks
    interleaved with max pooling layers, following the given 
    input/hidden/output number of channels

    Note: the `conv_params` and `pool_params` arguments should be tuples
    containing (kernel_size, stride, padding) to use with the respective layer
    '''
    assert len(hidden_channels) >= 1

    convs = []
    channels = [input_channels] + hidden_channels + [output_channels]
    conv_kernel_size, conv_stride, conv_padding = conv_params
    pool_kernel_size, pool_stride, pool_padding = pool_params
    for i in range(1, len(channels)):
        block = conv_bn_act(
            channels[i - 1], channels[i],
            kernel_size=conv_kernel_size, stride=conv_stride,
            padding=conv_padding, nonlinearity=nonlinearity
        )
        # Add pooling once every two layers
        if i % 2 == 0:
            block += [
                nn.MaxPool2d(
                    kernel_size=pool_kernel_size,
                    stride=pool_stride,
                    padding=pool_padding
                )
            ]
        convs.extend(block)

    return nn.Sequential(*convs)


def conv_block_output_size(modules, input_width, input_height):
    '''
    Given a sequence of PyTorch modules (e.g. Python list, PyTorch Sequential/ModuleList)
    containing convolution related layers (currently only Conv2d and MaxPool2d are supported),
    returns the output size of the input tensor, after it passes through all the given layers
    '''
    output_width, output_height = input_width, input_height
    for module in modules:
        if type(module) in (nn.Conv2d, nn.MaxPool2d):
            if type(module) == nn.Conv2d:
                kernel_size, stride, padding, dilation = get_conv2d_params(
                    module
                )
            elif type(module) == nn.MaxPool2d:
                kernel_size, stride, padding, dilation = get_maxpool2d_params(
                    module
                )
            kernel_size_h, kernel_size_w = kernel_size
            stride_h, stride_w = stride
            padding_h, padding_w = padding
            dilation_h, dilation_w = dilation
            output_width = np.floor((
                output_width + 2 * padding_w -
                dilation_w * (kernel_size_w - 1) - 1
            ) / stride_w + 1)
            output_height = np.floor((
                output_height + 2 * padding_h -
                dilation_h * (kernel_size_h - 1) - 1
            ) / stride_h + 1)
    return int(output_width), int(output_height)


def get_conv2d_params(conv):
    '''
    Return kernel size, stride, padding and dilation for a Conv2d layer
    '''
    return (
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation
    )


def get_maxpool2d_params(pool):
    '''
    Return kernel size, stride, padding and dilation for a MaxPool2d layer
    '''
    return (
        (pool.kernel_size, pool.kernel_size),
        (pool.stride, pool.stride),
        (pool.padding, pool.padding),
        (pool.dilation, pool.dilation)
    )
