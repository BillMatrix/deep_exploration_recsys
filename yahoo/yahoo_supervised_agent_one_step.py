from yahoo_environment import YahooFeedUnit
from typing import List, Dict, Tuple
import torch
import torch.optim as optim
import itertools
import math
import copy
import numpy as np
from helper import SupervisedTransition, SupervisedMemory, MLP

has_gpu = torch.cuda.is_available()
device = torch.device("cuda" if has_gpu else "cpu")


class YahooSupervisedAgentOneStep():
    def __init__(self,
                 initial_feed_candidates,
                 user_features,
                 feed_counts: int,
                 agent_name: str,
                 feed_feature_count = 6,
                 user_feature_count = 6,
                 model_dims = [50, 25],
                 batch_size: int = 128,
                 interest_unknown: bool = False,
                 boltzmann: bool = True):
        self.initial_feed_candidates = initial_feed_candidates
        self.current_feed_candidates = initial_feed_candidates
        self.user_features = user_features
        self.feed_counts = feed_counts
        self.agent_name = agent_name

        self.cum_rewards: float = 0.
        self.rewards = []
        self.actions = []
        self.training_data = []
        self.feed_feature_count = feed_feature_count
        self.user_feature_count = user_feature_count
        self.num_features: int = feed_feature_count + user_feature_count
        self.buffer: SupervisedMemory = SupervisedMemory(100000)

        self.model_dims: List[int] = [self.num_features] + model_dims + [1]
        self.model = MLP(self.model_dims).double()
        self.model.initialize()
        self.model.to(device)

        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.boltzmann: bool = boltzmann
        self.epsilon: float = 0.05
        self.batch_size: int = batch_size
        self.gamma = 0.99
        self.running_loss = 0.0
        self.history_actions = []
        self.latest_feature = None
        self.current_feed = 0
        self.cum_reward_history: List[float] = []

    def choose_action(self):
        available_actions = [candidate.features for candidate in self.current_feed_candidates]

        features = np.array([-1. for _ in range(self.num_features)])
        features[-self.user_feature_count:] = self.user_features

        candidate_features = []
        for f in available_actions:
            candidate_feature = np.copy(features)
            candidate_feature[
                :self.feed_feature_count
            ] = f
            candidate_features.append(candidate_feature)
        candidate_features = np.array(candidate_features)

        with torch.no_grad():
            outcomes = self.model(
                torch.tensor(candidate_features, dtype=torch.double).to(device)
            )

            _, best_index = torch.max(outcomes, 0)
            best_index = best_index.item()

            if self.boltzmann and np.random.rand() < 0.05:
                best_index = np.random.choice(len(outcomes.cpu().numpy().tolist()))

            best_action = self.current_feed_candidates[best_index]
            self.latest_feature = candidate_features[best_index]
            self.history_actions.append(best_action.features)

            self.current_feed += 1
            return best_action

    def update_buffer(
        self,
        scroll: bool,
        reward: int,
        new_batch
    ):
        self.cum_rewards += reward
        self.rewards += [reward]
        self.training_data += [self.latest_feature]
        self.current_feed_candidates = new_batch

    def learn_from_buffer(self):
        for i, data in enumerate(self.training_data):
            self.buffer.push(
                torch.tensor([data], dtype=torch.double).to(device),
                torch.tensor([[self.rewards[i]]], dtype=torch.double).to(device),
            )

        if len(self.buffer) < self.batch_size:
            return

        loss_ensemble = 0.
        for _ in range(10):
            transitions = self.buffer.sample(self.batch_size)
            batch = SupervisedTransition(*zip(*transitions))
            state_batch = torch.cat(batch.feature)
            lifetime_value_batch = torch.cat(batch.lifetime_value)

            predicted_lifetime_value = self.model(state_batch)
            loss = self.loss_fn(predicted_lifetime_value, lifetime_value_batch)
            loss_ensemble += loss.item()

            self.optimizer.zero_grad()
            loss.backward()

            for param in self.model.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

        self.running_loss = 0.8 * self.running_loss + 0.2 * loss_ensemble

    def reset(self, user_features, initial_feeds, user_embedding):
        self.cum_rewards: float = 0.
        self.interest_level = 0.
        self.latest_feature = None
        self.history_actions = []
        self.rewards = []
        self.actions = []
        self.training_data = []
        self.cum_reward_history.append(self.cum_rewards)
        self.current_feed = 0
        self.current_feed_candidates = initial_feeds
        self.user_features = user_features
