import numpy as np
import torch
import torch.nn as nn


def get_linear(input_size, output_size, hidden_sizes, nonlinearity="tanh"):
    '''
    Returns a PyTorch Sequential object containing FC layers with
    non-linear activation functions, by following the given input/hidden/output sizes
    '''
    assert len(hidden_sizes) >= 1
    nl = nn.ReLU() if nonlinearity == "relu" else nn.Tanh()
    fc = [nn.Linear(input_size, hidden_sizes[0]), nl]
    for i in range(1, len(hidden_sizes)):
        fc.extend([nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]), nl])
    fc.extend([nn.Linear(hidden_sizes[-1], output_size)])
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
        nn.ReLU() if nonlinearity == "relu" else nn.Tanh()
    ]


def conv_bn_act_maxpool(input_channels, output_channels, kernel_size=3,
                        stride=1, padding=0, nonlinearity="relu"):
    '''
    Returns a block composed by a convolutional layer and a batch norm one,
    followed by a non-linearity (e.g. ReLU or Tanh), with a final max pooling layer
    '''
    return (
        conv_bn_act(
            input_channels, output_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, nonlinearity=nonlinearity
        ) +
        [nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)]
    )


def get_conv(input_channels, output_channels, hidden_channels,
             kernel_size=3, stride=1, padding=0, nonlinearity="relu", pool=False):
    '''
    Returns a PyTorch Sequential object containing `conv_bn_act` or `conv_bn_act_pool` blocks,
    by following the given input/hidden/output number of channels
    '''
    assert len(hidden_channels) >= 1
    conv_block = conv_bn_act if not pool else conv_bn_act_maxpool
    conv = conv_block(
        input_channels, hidden_channels[0], kernel_size=kernel_size,
        stride=stride, padding=padding, nonlinearity=nonlinearity
    )
    for i in range(1, len(hidden_channels)):
        conv.extend(
            conv_block(
                hidden_channels[i - 1], hidden_channels[i],
                kernel_size=kernel_size, stride=stride, padding=padding,
                nonlinearity=nonlinearity
            )
        )
    conv.extend(
        conv_block(
            hidden_channels[-1], output_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, nonlinearity=nonlinearity
        )
    )
    return nn.Sequential(*conv)


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
