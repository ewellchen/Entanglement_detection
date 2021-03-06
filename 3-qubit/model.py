#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Construct the computational graph of model for training. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn

from complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear, ComplexConvTranspose2d
from complexFunctions import complex_relu, complex_max_pool2d


class Encoder_r(nn.Module):
    def __init__(self, k1=2, c1=40, k2=2, c2=100, k3=3, c3=3, d1=96, d2=10):
        super(Encoder_r, self).__init__()
        self.conv1 = ComplexConv2d(1, c1, k1, 1, padding=0)
        self.bn1 = ComplexBatchNorm2d(c1)
        self.conv2 = ComplexConv2d(c1, c2, k2, 1, padding=0)
        self.bn2 = ComplexBatchNorm2d(c2)
        self.conv3 = ComplexConv2d(c2, c3, k3, 1, padding=0)
        self.fc1 = ComplexLinear(2 * 2 * c3, d1)
        self.fc2 = ComplexLinear(d1, d2)
        self.c3 = c3

    def forward(self, xr, xi):
        # xr = x[:, :, :, :, 0]
        # # imaginary part to zero
        # xi = x[:, :, :, :, 1]
        xr, xi = self.conv1(xr, xi)
        xr, xi = complex_relu(xr, xi)
        # xr, xi = complex_max_pool2d(xr, xi, 2, 2)

        xr, xi = self.bn1(xr, xi)
        xr, xi = self.conv2(xr, xi)
        xr, xi = complex_relu(xr, xi)
        # xr, xi = complex_max_pool2d(xr, xi, 2, 2)

        xr, xi = self.bn2(xr, xi)
        xr, xi = self.conv3(xr, xi)
        xr, xi = complex_relu(xr, xi)
        # xr, xi = complex_max_pool2d(xr, xi, 2, 2)

        xr = xr.view(-1, 2 * 2 * self.c3)
        xi = xi.view(-1, 2 * 2 * self.c3)
        xr, xi = self.fc1(xr, xi)
        xr, xi = complex_relu(xr, xi)
        xr, xi = self.fc2(xr, xi)
        # take the absolute value as output
        # x = torch.sqrt(torch.pow(xr, 2) + torch.pow(xi, 2))
        return xr, xi


class Generator(nn.Module):

    def __init__(self, k1=2, c1=40, k2=2, c2=100, k3=3, c3=3, d1=96, d2=10):
        super(Generator, self).__init__()
        self.convt1 = ComplexConvTranspose2d(c1, 1, k1, 1, padding=0)
        self.bn1 = ComplexBatchNorm2d(c1)
        self.convt2 = ComplexConvTranspose2d(c2, c1, k2, 1, padding=0)
        self.bn2 = ComplexBatchNorm2d(c2)
        self.convt3 = ComplexConvTranspose2d(c3, c2, k3, 1, padding=0)  # k = 2,p' = k - 1
        self.fc1 = ComplexLinear(d1, 2 * 2 * c3)
        self.fc2 = ComplexLinear(d2, d1)
        self.c3 = c3

    def forward(self, xr, xi):
        # imaginary part to zero
        xr, xi = self.fc2(xr, xi)
        xr, xi = complex_relu(xr, xi)

        # xr, xi = complex_max_pool2d(xr, xi, 2, 2)
        xr, xi = self.fc1(xr, xi)
        xr = xr.view(-1, self.c3, 2, 2)
        xi = xi.view(-1, self.c3, 2, 2)
        xr, xi = complex_relu(xr, xi)
        xr, xi = self.convt3(xr, xi)
        xr, xi = self.bn2(xr, xi)

        xr, xi = complex_relu(xr, xi)
        xr, xi = self.convt2(xr, xi)
        xr, xi = self.bn1(xr, xi)

        # xr, xi = complex_max_pool2d(xr, xi, 2, 2)
        xr, xi = complex_relu(xr, xi)
        xr, xi = self.convt1(xr, xi)

        return xr, xi


class Encoder_f(nn.Module):

    def __init__(self, k1=2, c1=40, k2=2, c2=100, k3=3, c3=3, d1=96, d2=10):
        super(Encoder_f, self).__init__()
        self.conv1 = ComplexConv2d(1, c1, k1, 1, padding=0)
        self.bn1 = ComplexBatchNorm2d(c1)
        self.conv2 = ComplexConv2d(c1, c2, k2, 1, padding=0)
        self.bn2 = ComplexBatchNorm2d(c2)
        self.conv3 = ComplexConv2d(c2, c3, k3, 1, padding=0)
        self.fc1 = ComplexLinear(2 * 2 * c3, d1)
        self.fc2 = ComplexLinear(d1, d2)
        self.c3 = c3

    def forward(self, xr, xi):
        # xr = x[:, :, :, :, 0]
        # # imaginary part to zero
        # xi = x[:, :, :, :, 1]
        xr, xi = self.conv1(xr, xi)
        xr, xi = complex_relu(xr, xi)
        # xr, xi = complex_max_pool2d(xr, xi, 2, 2)

        xr, xi = self.bn1(xr, xi)
        xr, xi = self.conv2(xr, xi)
        xr, xi = complex_relu(xr, xi)
        # xr, xi = complex_max_pool2d(xr, xi, 2, 2)

        xr, xi = self.bn2(xr, xi)
        xr, xi = self.conv3(xr, xi)
        xr, xi = complex_relu(xr, xi)
        # xr, xi = complex_max_pool2d(xr, xi, 2, 2)

        xr = xr.view(-1, 2 * 2 * self.c3)
        xi = xi.view(-1, 2 * 2 * self.c3)
        xr, xi = self.fc1(xr, xi)
        xr, xi = complex_relu(xr, xi)
        xr, xi = self.fc2(xr, xi)
        # take the absolute value as output
        # x = torch.sqrt(torch.pow(xr, 2) + torch.pow(xi, 2))
        return xr, xi


class Discriminator(nn.Module):

    def __init__(self, k1=2, c1=40, k2=2, c2=100, k3=3, c3=3, d1=96, d2=10):
        super(Discriminator, self).__init__()
        self.conv1 = ComplexConv2d(1, c1, k1, 1, padding=0)
        self.bn1 = ComplexBatchNorm2d(c1)
        self.conv2 = ComplexConv2d(c1, c2, k2, 1, padding=0)
        self.bn2 = ComplexBatchNorm2d(c2)
        self.conv3 = ComplexConv2d(c2, c3, k3, 1, padding=0)
        self.fc1 = ComplexLinear(2 * 2 * c3, d1)
        self.fc2 = ComplexLinear(d1, d2)
        self.c3 = c3

    def forward(self, xr, xi):
        # xr = x[:, :, :, :, 0]
        # # imaginary part to zero
        # xi = x[:, :, :, :, 1]
        xr, xi = self.conv1(xr, xi)
        xr, xi = complex_relu(xr, xi)
        # xr, xi = complex_max_pool2d(xr, xi, 2, 2)

        xr, xi = self.bn1(xr, xi)
        xr, xi = self.conv2(xr, xi)
        xr, xi = complex_relu(xr, xi)
        # xr, xi = complex_max_pool2d(xr, xi, 2, 2)

        xr, xi = self.bn2(xr, xi)
        xr, xi = self.conv3(xr, xi)
        xr, xi = complex_relu(xr, xi)
        # xr, xi = complex_max_pool2d(xr, xi, 2, 2)

        xr = xr.view(-1, 2 * 2 * self.c3)
        xi = xi.view(-1, 2 * 2 * self.c3)
        xr, xi = self.fc1(xr, xi)
        xr, xi = complex_relu(xr, xi)
        xr, xi = self.fc2(xr, xi)
        x = torch.sqrt(torch.pow(xr, 2) + torch.pow(xi, 2))
        return x
