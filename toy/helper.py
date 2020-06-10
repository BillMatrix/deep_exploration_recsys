from math import exp
import torch
import torch.nn as nn
import torch.nn.functional as f
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
                m.bias.data.fill_(0.1)
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
                        ('state', 'action', 'reward', 'next_state'))


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

SupervisedTransition = namedtuple('SupervisedTransition', ('feature', 'lifetime_value', 'actions'))


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


class HyperNetwork(nn.Module):
    def __init__(self, dims, z_dim):
        super(HyperNetwork, self).__init__()
        self.z_dim = z_dim
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))

        def initialize_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.0)
        self.apply(initialize_weights)

    def forward(self, z):
        for l in self.layers[:-1]:
            z = nn.functional.relu(l(z))
        z = self.layers[-1](z)
        return z


class HyperDQN(nn.Module):
    def __init__(self, hyper_dims, model_dims, z_dim):
        super(HyperDQN, self).__init__()
        self.z_dim = z_dim
        self.hyper_dims = hyper_dims
        self.hyper_networks = []
        self.bias_hyper_networks = []
        self.model_dims = model_dims
        for i in range(len(model_dims) - 1):
            parameter_size = model_dims[i] * model_dims[i + 1]
            hyper_network = HyperNetwork(hyper_dims + [parameter_size], z_dim)
            hyper_network = hyper_network.double()
            self.hyper_networks.append(hyper_network)
            bias_hyper_network = HyperNetwork(hyper_dims + [model_dims[i + 1]], z_dim)
            bias_hyper_network = bias_hyper_network.double()
            self.bias_hyper_networks.append(bias_hyper_network)

    def forward(self, x, z):
        # x = x.view([x.shape[0], 1, x.shape[1]])
        for i in range(len(self.hyper_networks)):
            # print('z shape: ', z.shape)
            weights = self.hyper_networks[i](z)
            # print('weights: ', weights.shape)
            # print('x shape: ', x.shape)
            weights = weights.view(
                (-1, self.model_dims[i], self.model_dims[i + 1])
            )
            # print('reshaped weights: ', weights.shape)
            bias = self.bias_hyper_networks[i](z)
            # print('bias shape: ', bias.shape)
            # print('prod shape: ', torch.einsum('bj,bjk->bk', x, weights).shape)
            # print('prod reshape shape: ', x.matmul(weights).view([x.shape[0], -1]).shape)
            x = f.relu(torch.einsum('bj,bjk->bk', x, weights) + bias)

        return x

    def parameters(self, recurse: bool = True):
        parameters = []
        for net in self.hyper_networks:
            parameters += list(net.parameters())
        for net in self.bias_hyper_networks:
            parameters += list(net.parameters())
        return parameters



class HyperDQNWithPrior(nn.Module):
    def __init__(self, hyper_dims, model_dims, z_dim, f_prior, scale=5):
        """
        :param dimensions: dimensions of the neural network
        prior network with immutable weights and
        difference network whose weights will be learnt
        """

        super(HyperDQNWithPrior, self).__init__()
        self.f_diff = HyperDQN(hyper_dims, model_dims, z_dim)
        self.f_prior = f_prior
        self.scale = scale

    def forward(self, x, z):
        """
        :param x: input to the network
        :return: computes f_diff(x) + f_prior(x)
        performs forward pass of the network
        """
        # print('f_diff: ', self.f_diff(x, z).shape)
        # print('f_prior: ', self.f_prior(x).shape)
        return (
            self.f_diff(x, z)
            # + self.scale * self.f_prior(x)
        )

    def initialize(self):
        self.f_diff.initialize()

    def parameters(self, recurse: bool = True):
        """
        :param recurse: bool Recursive or not
        :return: all the parameters of the network that are mutable or learnable
        """
        return self.f_diff.parameters(recurse)

    def set_prior(self, f_prior):
        self.f_prior = f_prior
