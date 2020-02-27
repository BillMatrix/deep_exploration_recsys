from math import exp
import torch
import torch.nn as nn
from collections import namedtuple
import random


def sigmoid(x):
    return 1. / (1. + exp(-x / 3))


class MLP(nn.Module):
    def __init__(self, dimensions):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(dimensions)-1):
            self.layers.append(nn.Linear(dimensions[i], dimensions[i+1]))

    def forward(self, x):
        for l in self.layers[:-1]:
            x = nn.functional.relu(l(x))
        x = self.layers[-1](x)
        return x

    def initialize(self):
        """
        Initialize weights using Glorot initialization
        or also known as Xavier initialization
        """
        def initialize_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        self.apply(initialize_weights)


class DQNWithPrior(nn.Module):
    def __init__(self,dimensions, f_prior, scale=5):
        """
        :param dimensions: dimensions of the neural network
        prior network with immutable weights and
        difference network whose weights will be learnt
        """

        super(DQNWithPrior,self).__init__()
        self.f_diff = MLP(dimensions)
        self.f_prior = f_prior
        self.scale = scale

    def forward(self, x):
        """
        :param x: input to the network
        :return: computes f_diff(x) + f_prior(x)
        performs forward pass of the network
        """
        return self.f_diff(x) + self.scale * self.f_prior(x)

    def initialize(self):
        self.f_diff.initialize()

    def parameters(self, recurse: bool = True):
        """
        :param recurse: bool Recursive or not
        :return: all the parameters of the network that are mutable or learnable
        """
        return self.f_diff.parameters(recurse)


Transition = namedtuple('Transition',
                        ('state', 'reward', 'next_state'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

FeedTransition = namedtuple('FeedTransition',
                        ('state', 'action', 'reward', 'next_state', 'next_available_actions'))


class FeedReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = FeedTransition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

SupervisedTransition = namedtuple('SupervisedTransition', ('feature', 'lifetime_value'))


class SupervisedMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = SupervisedTransition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
