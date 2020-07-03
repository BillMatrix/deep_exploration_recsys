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


NCFTransition = namedtuple('NCFTransition', (
    'feature', 'user_embedding', 'feed_embedding', 'lifetime_value'))


class NCFMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = NCFTransition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


NCFReplayTransition = namedtuple('NCFReplayTransition', (
    'state', 'user_embedding', 'cur_feed_embedding',
    'reward', 'next_state', 'next_user_embeddings',
    'next_feed_embeddings'
))


class NCFReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = NCFReplayTransition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class NCF(nn.Module):
    def __init__(
        self,
        user_num,
        item_num,
        dense_feature_size,
        factor_num,
        num_layers,
        dropout,
        model,
        GMF_model=None,
        MLP_model=None,
    ):
        super(NCF, self).__init__()
        self.dropout = dropout
        self.model = model
        self.GMF_model = GMF_model
        self.MLP_model = MLP_model

        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        self.embed_item_GMF = nn.Embedding(item_num, factor_num)
        self.embed_user_MLP = nn.Embedding(
            user_num, factor_num * (2 ** (num_layers - 1))
        )
        self.embed_item_MLP = nn.Embedding(
            item_num, factor_num * (2 ** (num_layers - 1))
        )

        MLP_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            # MLP_modules.append(nn.Dropout(p=self.dropout))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        if self.model in ['MLP', 'GMF']:
            predict_size = factor_num + dense_feature_size
        else:
            predict_size = factor_num * 2 + dense_feature_size
        self.predict_layer = nn.Linear(predict_size, 1)

        self._init_weight_()

    def _init_weight_(self):
        if not self.model == 'NeuMF-pre':
            nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
            nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

            for m in self.MLP_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
            nn.init.kaiming_uniform_(
                self.predict_layer.weight, a=1, nonlinearity='sigmoid')

            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
        else:
            self.embed_user_GMF.weight.data.copy_(
                self.GMF_model.embed_user_GMF.weight
            )
            self.embed_item_GMF.weight.data.copy_(
                self.GMF_model.embed_item_GMF.weight
            )
            self.embed_user_MLP.weight.data.copy_(
                self.MLP_model.embed_user_MLP.weight
            )
            self.embed_item_MLP.weight.data.copy_(
                self.MLP_model.embed_user_MLP.weight
            )

            for (m1, m2) in zip(self.MLP_layers, self.MLP_model.MLP_layers):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)

            predict_weight = torch.cat([self.GMF_model.predict_layer.weight,
                self.MLP_model.predict_layer.weight], dim=1)
            precit_bias = self.GMF_model.predict_layer.bias + self.MLP_model.predict_layer.bias

            self.predict_layer.weight.data.copy_(0.5 * predict_weight)
            self.predict_layer.bias.data.copy_(0.5 * precit_bias)

    def forward(self, user, item, dense_features, join_dim=1):
        if not self.model == 'MLP':
            embed_user_GMF = self.embed_user_GMF(user)
            embed_item_GMF = self.embed_item_GMF(item)
            output_GMF = embed_user_GMF * embed_item_GMF
        if not self.model == 'GMF':
            embed_user_MLP = self.embed_user_MLP(user)
            embed_item_MLP = self.embed_item_MLP(item)
            interaction = torch.cat((embed_user_MLP, embed_item_MLP), join_dim)
            output_MLP = self.MLP_layers(interaction)

        if self.model == 'GMF':
            concat = output_GMF
        elif self.model == 'MLP':
            concat = output_MLP
        else:
            concat = torch.cat(
                (output_GMF, output_MLP, dense_features),
                join_dim,
            )

        return self.predict_layer(concat)


class NCFWithPrior(nn.Module):
    def __init__(self, f_diff, f_prior, scale=5):
        """
        :param dimensions: dimensions of the neural network
        prior network with immutable weights and
        difference network whose weights will be learnt
        """

        super(NCFWithPrior,self).__init__()
        self.f_diff = f_diff
        self.f_prior = f_prior
        self.scale = scale

    def forward(self, user, item, dense_features, join_dim=1):
        """
        :param x: input to the network
        :return: computes f_diff(x) + f_prior(x)
        performs forward pass of the network
        """
        return self.f_diff(user, item, dense_features, join_dim) + self.scale * self.f_prior(dense_features)

    def parameters(self, recurse: bool = True):
        """
        :param recurse: bool Recursive or not
        :return: all the parameters of the network that are mutable or learnable
        """
        return self.f_diff.parameters(recurse)
