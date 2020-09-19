#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from tools import mkdir
import argparse


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
                    s = self.generate_sub_state()
                    for number in range(self.space_number - 1):
                        s = np.reshape(np.einsum('ij,kl->ikjl', s, self.generate_sub_state()),
                                       [s.shape[0] * self.sub_dim, s.shape[1] * self.sub_dim])
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
    def __init__(self, name='ent_test_set', size=40000, sub_dim=2, space_number=3):
        self.name = name
        self.size = size
        self.sub_dim = sub_dim
        self.space_number = space_number
        self.v1 = [1, 0, 0, 0, 0, 0, 0, 0]
        self.v2 = [0, 0, 0, 0, -0.5, -0.5, 0.5, 0.5]
        self.v3 = [0, 0, -0.5, 0.5, 0, 0, -0.5, 0.5]
        self.v4 = [0, -0.5, 0, -0.5, 0, 0.5, 0, 0.5]
        p1 = np.outer(self.v1, self.v1)
        p2 = np.outer(self.v2, self.v2)
        p3 = np.outer(self.v3, self.v3)
        p4 = np.outer(self.v4, self.v4)
        self.ptile = (np.eye(8) - p1 - p2 - p3 - p4) / 4
        self.state = []

    def generate_sub_matrix(self):
        q1 = np.random.normal(0, 1, [self.sub_dim, self.sub_dim])
        q2 = np.random.normal(0, 1, [self.sub_dim, self.sub_dim])
        q = q1 + 1j * q2
        q = np.matrix(q)  # Create a random Complex matrix H.
        q = (q + q.H) / 2  # Generate GUE
        q = np.dot(q, q.H) / np.trace(np.dot(q, q.H))  # Make it positive, and normalize to unity.
        return q

    def generate(self):
        for s in range(self.size):
            sigma = []
            for i in range(self.space_number):
                q1 = np.random.normal(0, 1, [self.sub_dim, self.sub_dim])
                q2 = np.random.normal(0, 1, [self.sub_dim, self.sub_dim])
                q = q1 + 1j * q2
                sigma.append(q)  # Create a random Complex matrix H.
            Sigma = np.reshape(np.einsum('ij,kl,mn->ikmjln', sigma[0], sigma[1], sigma[2]),
                               [pow(self.sub_dim, self.space_number), pow(self.sub_dim, self.space_number)])
            Sigma = np.matrix(Sigma)
            transition = np.dot(np.dot(Sigma, self.ptile), Sigma.H)
            rho = transition / np.trace(transition)
            self.state.append(rho)
        State = np.array(self.state)
        shape = list(State.shape)
        shape.append(1)
        Set_r = np.reshape(np.real(State), shape)
        Set_i = np.reshape(np.imag(State), shape)
        Set_2 = np.concatenate((Set_r, Set_i), axis=-1)
        mkdir('./Data/')
        np.save('./Data/' + self.name + '.npy', Set_2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate quantum states for training and testing.')
    parser.add_argument('-type', dest='type', help='please enter the type of states.', default='separable')
    parser.add_argument('-test_number', dest='test_number', help='please enter the number for testing.', default=1000)
    parser.add_argument('-train_number', dest='train_number', help='please enter the number for training.', default=10000)
    parser.add_argument('-sub_dim', dest='sub_dim', help='please enter the dimension of sub states.', default=2)
    parser.add_argument('-number', dest='number', help='please enter the number of sub states.', default=2)

    args = parser.parse_args()

    Generate_separable_state(name='sep_train_set', size=args.train_number, sub_dim=args.sub_dim, space_number=args.number,
                             mix_number=20).generate()
    Generate_separable_state(name='sep_test_set', size=args.test_number, sub_dim=args.sub_dim, space_number=args.number,
                             mix_number=20).generate()
    Generate_entangled_state(name='ent_test_set', size=args.test_number, sub_dim=args.sub_dim).generate()
    print('Generating data done!')
