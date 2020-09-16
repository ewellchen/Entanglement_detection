#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Construct the tools. """


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import os
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print('Folder have been made!')


def l1_loss(input, target, test=False):
    """ L1 Loss without reduce flag.
    Args:
        input (FloatTensor): Input tensor
        target (FloatTensor): Output tensor
    Returns:
        [FloatTensor]: L1 distance between input and output
    """
    if test:
        return torch.mean(torch.abs(input[0] - target[0]) + torch.abs(input[1] - target[1]), dim=1, keepdim=False)
    else:
        return torch.mean(torch.abs(input[0] - target[0]) + torch.abs(input[1] - target[1]))


##
def l2_loss_real(input, target, size_average=True):
    """ L2 Loss without reduce flag.

    Args:
        input (FloatTensor): Input tensor
        target (FloatTensor): Output tensor

    Returns:
        [FloatTensor]: L2 distance between input and output
    """
    if size_average:
        return torch.mean(torch.pow((input - target), 2))
    else:
        return torch.pow((input - target), 2)


def l2_loss_complex(input, target, test=False):
    """ L2 Loss without reduce flag.

    Args:
        input (FloatTensor): Input tensor
        target (FloatTensor): Output tensor

    Returns:
        [FloatTensor]: L2 distance between input and output
    """
    if test:
        return torch.mean(torch.pow((input[0] - target[0]), 2) + torch.pow((input[1] - target[1]), 2), dim=1,
                          keepdim=False)
    else:
        return torch.mean(torch.pow((input[0] - target[0]), 2) + torch.pow((input[1] - target[1]), 2))


def roc(labels, scores, number, saveto=None):
    """Compute ROC curve and ROC area for each class"""
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # True/False Positive Rates.
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    print('AUC: %0.4f' % (roc_auc))
    # Equal Error Rate
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    print('EER: %0.4f' % (eer))
    if saveto:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.4f, EER = %0.4f)' % (roc_auc, eer))

        plt.plot([eer], [1 - eer], marker='o', markersize=5, color="navy")
        plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(saveto, 'ROC_%d.png' % (number)))
        plt.close()

    return roc_auc, eer
