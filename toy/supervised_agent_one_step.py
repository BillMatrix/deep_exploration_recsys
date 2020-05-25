from environment import FeedUnit
from typing import List, Dict, Tuple
from agent import Agent
import torch
import torch.optim as optim
import itertools
import math
import copy
import numpy as np
from scipy.special import softmax
from helper import MLP, SupervisedTransition, SupervisedMemory

has_gpu = torch.cuda.is_available()
device = torch.device("cpu")


class SupervisedAgentOneStep(Agent):
    def __init__(
            self,
            feed_units: List[int],
            agent_name: str,
            model_dims: List[int] = [20],
            lr: float = 1e-3,
            boltzmann: bool = False,
            epsilon: float = 0.05,
            batch_size: int = 128,
    ):
        self.feed_units = copy.deepcopy(feed_units)
        self.agent_name = agent_name
        self.interest_level = 0

        self.cum_rewards: float = 0.
        self.num_features: int = len(feed_units)
        self.training_data = []
        self.buffer: SupervisedMemory = SupervisedMemory(100000)

        self.model_dims: List[int] = [self.num_features] + model_dims + [2]
        self.model = MLP(self.model_dims).double()
        self.model.initialize()
        self.model.to(device)

        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.boltzmann: bool = boltzmann
        self.epsilon: float = epsilon
        self.batch_size: int = batch_size
        self.gamma = 0.99
        self.running_loss = 0.0
        self.history_unit_indices: List[int] = []
        self.latest_feature = None
        self.latest_action = None
        self.current_feed = 0
        self.cum_reward_history: List[float] = []
        self.rewards: List[float] = []
        self.actions = []

    def choose_action(self):
        available_actions = [0, 1]

        features: List[float] = [0. for _ in range(self.num_features)]
        features[self.current_feed] = 1.
        # for index in range(self.current_feed):
        #     features[index] = 0.
        # for index in self.history_unit_indices:
        #     features[index] = 1.

#         base_feature.append(self.interest_level)
        with torch.no_grad():
            outcomes = self.model(
                torch.tensor(features, dtype=torch.double)
            )

            _, best_index = torch.max(outcomes, 0)
            best_index = best_index.item()

            best_action = [available_actions[best_index]]
            self.latest_feature = features
            self.latest_action = best_action
            if best_action[0] == 1:
                self.history_unit_indices.append(self.current_feed)

            self.current_feed += 1

            if np.random.rand() < self.epsilon:
                return np.random.randint(2)
#             print(best_action)
            return best_action[0]

    def update_buffer(
        self,
        scroll: bool,
        reward: int,
    ):
#         print(reward)
        self.cum_rewards += reward
        self.rewards += [reward]
        self.training_data += [self.latest_feature]
        self.actions += [self.latest_action]
        # self.current_feed += 1

    def learn_from_buffer(self):
        # print(self.actions)
        for i, data in enumerate(self.training_data):
            self.buffer.push(
                torch.tensor([data], dtype=torch.double),
                torch.tensor([[self.rewards[i]]], dtype=torch.double),
                torch.tensor([self.actions[i]], dtype=torch.long),
            )

        if len(self.buffer) < self.batch_size:
            return

        loss_ensemble = 0.
        for _ in range(10):
            transitions = self.buffer.sample(self.batch_size)
            batch = SupervisedTransition(*zip(*transitions))
            state_batch = torch.cat(batch.feature)
            action_batch = torch.cat(batch.actions)
            lifetime_value_batch = torch.cat(batch.lifetime_value)

            predicted_lifetime_value = self.model(state_batch).gather(1, action_batch)
            loss = self.loss_fn(predicted_lifetime_value, lifetime_value_batch)
            loss_ensemble += loss.item()

            self.optimizer.zero_grad()
            loss.backward()

            for param in self.model.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

        self.running_loss = 0.8 * self.running_loss + 0.2 * loss_ensemble

    def reset(self):
        self.cum_rewards: float = 0.
        self.interest_level = 0.
        self.latest_feature = None
        self.latest_action = None
        self.history_unit_indices = []
        self.cum_reward_history.append(self.cum_rewards)
        self.current_loc = [0, 0]
        self.current_feed = 0
        self.rewards = []
        self.actions = []
        self.training_data = []
