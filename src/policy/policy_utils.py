import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMSELoss(nn.Module):
    '''
    MSE loss with masked inputs/targets
    '''

    def __init__(self, reduction='mean'):
        super(MaskedMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target, mask=None):
        if mask is None:
            return F.mse_loss(input, target, reduction=self.reduction)

        flattened_mask = torch.flatten(mask)
        diff = ((
            torch.flatten(input) - torch.flatten(target)
        ) ** 2.0) * flattened_mask
        mask_sum = (
            torch.sum(flattened_mask)
            if self.reduction == 'mean'
            else 1.0
        )
        return torch.sum(diff) / mask_sum


class MaskedHuberLoss(nn.Module):
    '''
    Huber loss with masked inputs/targets
    '''

    def __init__(self, reduction='mean', beta=1.0):
        super(MaskedHuberLoss, self).__init__()
        self.reduction = reduction
        self.beta = float(beta)

    def forward(self, input, target, mask=None):
        if mask is None:
            return F.smooth_l1_loss(
                input, target, reduction=self.reduction, beta=self.beta
            )

        flattened_mask = torch.flatten(mask)
        errors = torch.abs(torch.flatten(input) - torch.flatten(target))
        diff = torch.where(
            errors < self.beta,
            flattened_mask * (0.5 * (errors ** 2) / self.beta),
            flattened_mask * (errors - 0.5 * self.beta)
        )
        mask_sum = (
            torch.sum(flattened_mask)
            if self.reduction == 'mean'
            else 1.0
        )
        return torch.sum(diff) / mask_sum


class Sequential(nn.Sequential):
    '''
    Extension of the PyTorch Sequential module,
    to handle a variable number of arguments
    '''

    def forward(self, input, *args, **kwargs):
        for module in self:
            input = module(input, *args, **kwargs)
        return input


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
