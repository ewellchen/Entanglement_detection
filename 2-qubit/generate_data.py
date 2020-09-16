#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from tools import mkdir


# 1. Create a random Complex matrix H.
# 2. Make it positive, and normalize to unity.

class Generate_separable_state(object):
    def __init__(self, name='sep_train_set', size=10000, sub_dim=2, space_number=2, mix_number=10):
        self.name = name
        self.size = size
        self.sub_dim = sub_dim
        self.space_number = space_number
        self.mix_number = mix_number
        self.set = []

    def generate_sub_state(self):
        q1 = np.random.normal(0, 1, [self.sub_dim, self.sub_dim])
        q2 = np.random.normal(0, 1, [self.sub_dim, self.sub_dim])
        q = q1 + 1j * q2
        q = np.matrix(q)  # Create a random Complex matrix H.
        q = (q + q.H) / 2  # Generate GUE
        q = np.dot(q, q.H) / np.trace(np.dot(q, q.H))  # Make it positive, and normalize to unity.
        return q

    def generate(self):
        for i in range(int(self.size / self.mix_number)):
            state = []
            for j in range(self.mix_number):
                s = 0
                if self.space_number > 1:
                    for number in range(self.space_number - 1):
                        s = np.reshape(np.einsum('ij,kl->ikjl', self.generate_sub_state(), self.generate_sub_state()),
                                       [self.sub_dim ** self.space_number, self.sub_dim ** self.space_number])
                    state.append(s)
            for j in range(self.mix_number):
                weight = np.random.random([j + 1])
                weight = weight / np.sum(weight)
                mix = np.zeros([self.sub_dim ** self.space_number, self.sub_dim ** self.space_number])
                for k in range(j + 1):
                    mix = mix + weight[k] * state[k]
                self.set.append(mix)
        Set = np.array(self.set)
        shape = list(Set.shape)
        shape.append(1)
        Set_r = np.reshape(np.real(Set), shape)
        Set_i = np.reshape(np.imag(Set), shape)
        Set_2 = np.concatenate((Set_r, Set_i), axis=-1)
        mkdir('./Data/')
        np.save('./Data/' + self.name + '.npy', Set_2)


class Generate_entangled_state(object):
    def __init__(self, name='ent_test_set', size=10000, dim=4):
        self.name = name
        self.size = size
        self.dim = dim
        self.state = []

    def ppt_criterion(self, rho, dims, mask):
        rho = np.array(rho)
        shape = rho.shape
        nsys = len(mask)
        pt_dims = np.arange(2 * nsys).reshape(2, nsys).T
        pt_idx = np.concatenate([[pt_dims[n, mask[n]] for n in range(nsys)],
                                 [pt_dims[n, 1 - mask[n]] for n in range(nsys)]])

        data = rho.reshape(
            np.array(dims).flatten()).transpose(pt_idx).reshape(shape)
        return np.all(np.linalg.eigvals(data) > 0)

    def generate(self):
        sum = 0
        while sum < self.size:
            s1 = np.random.normal(0, 1, [self.dim, self.dim])
            s2 = np.random.normal(0, 1, [self.dim, self.dim])
            s = s1 + 1j * s2
            s = np.matrix(s)  # Create a random Complex matrix H.
            s = (s + s.H) / 2  # Generate GUE
            s = np.dot(s, s.H) / np.trace(np.dot(s, s.H))  # Make it positive, and normalize to unity.
            if self.ppt_criterion(s, dims=[[2, 2], [2, 2]], mask=[0, 1]):
                a = 1
            else:
                self.state.append(s)
                sum += 1
        Set = np.array(self.state)
        shape = list(Set.shape)
        shape.append(1)
        Set_r = np.reshape(np.real(Set), shape)
        Set_i = np.reshape(np.imag(Set), shape)
        Set_2 = np.concatenate((Set_r, Set_i), axis=-1)
        mkdir('./Data/')
        np.save('./Data/' + self.name + '.npy', Set_2)


if __name__ == '__main__':
    set_name = 'mix'
    train_size = 160000
    test_size = 40000
    Generate_separable_state(name='sep_train_set', size=train_size, sub_dim=2, space_number=2, mix_number=10).generate()
    Generate_separable_state(name='sep_test_set', size=test_size, sub_dim=2, space_number=2, mix_number=10).generate()
    Generate_entangled_state(name='ent_test_set', size=test_size, dim=4).generate()

    print('Generating data is finished!')
