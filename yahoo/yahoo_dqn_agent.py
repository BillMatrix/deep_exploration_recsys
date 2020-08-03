from yahoo_environment import YahooFeedUnit
from typing import List, Dict, Tuple
import torch
import torch.optim as optim
import itertools
import math
import copy
import numpy as np
from scipy.special import softmax
from helper import MLP, Transition, ReplayMemory

has_gpu = torch.cuda.is_available()
device = torch.device("cuda" if has_gpu else "cpu")


class YahooDQNAgent():
    def __init__(
            self,
            initial_feed_candidates,
            user_features,
            feed_counts,
            agent_name: str,
            feed_feature_count = 6,
            user_feature_count = 6,
            model_dims: List[int] = [50, 25],
            lr: float = 1e-3,
            boltzmann: bool = True,
            epsilon: float = 0.05,
            batch_size: int = 128,
    ):
        self.initial_feed_candidates = initial_feed_candidates
        self.current_feed_candidates = initial_feed_candidates
        self.user_features = user_features
        self.feed_counts = feed_counts
        self.agent_name = agent_name
        self.interest_level = 0

        self.cum_rewards: float = 0.
        self.feed_feature_count = feed_feature_count
        self.user_feature_count = user_feature_count
        self.num_features = feed_counts * feed_feature_count + feed_feature_count + user_feature_count
        self.training_data: ReplayMemory = ReplayMemory(100000)

        self.model_dims: List[int] = [self.num_features] + model_dims + [1]
        self.model = MLP(self.model_dims).double()
        self.model.initialize()
        self.model.to(device)

        self.target_net = MLP(self.model_dims).double().to(device)
        self.target_net.load_state_dict(self.model.state_dict())
        self.target_net.eval()

        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.boltzmann: bool = boltzmann
        self.epsilon: float = epsilon
        self.batch_size: int = batch_size
        self.gamma = 0.99
        self.running_loss = 0.0
        self.history_actions = []
        self.latest_feature = None
        self.current_feed = 0
        self.cum_reward_history: List[float] = []

    def choose_action(self):
        available_actions = [candidate.features for candidate in self.current_feed_candidates]

        features = [-1. for _ in range(self.num_features)]
        for index, action in enumerate(self.history_actions):
            features[index * self.feed_feature_count:(index + 1) * self.feed_feature_count] = action
        features[-self.user_feature_count:] = self.user_features

        candidate_features = []
        for f in available_actions:
            candidate_feature = np.copy(features)
            candidate_feature[
                self.feed_counts * self.feed_feature_count:(self.feed_counts + 1) * self.feed_feature_count
            ] = f
            candidate_features.append(candidate_feature)
        candidate_features = np.array(candidate_features)

#         base_feature.append(self.interest_level)
        with torch.no_grad():
            outcomes = self.model(
                torch.tensor(candidate_features, dtype=torch.double).to(device)
            )

            _, best_index = torch.max(outcomes, 0)
            best_index = best_index.item()

            if self.boltzmann:
                outcomes = outcomes / 0.05
                best_index = np.random.choice(
                    len(available_actions),
                    p=torch.nn.functional.softmax(outcomes.reshape((len(available_actions))), dim=0).cpu().numpy()
                )
            elif np.random.rand() < 0.05:
                best_index = np.random.choice(len(available_actions))

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
#         print(reward)
        self.cum_rewards += reward
        self.current_feed_candidates = new_batch
        if not scroll:
            self.training_data.push(
                torch.tensor([self.latest_feature], dtype=torch.double).to(device),
                torch.tensor([reward], dtype=torch.double).to(device),
                None,
            )
            return

        available_actions = [candidate.features for candidate in self.current_feed_candidates]
        features: List[float] = [-1. for _ in range(self.num_features)]
        for index, action in enumerate(self.history_actions):
            features[index * self.feed_feature_count:(index + 1) * self.feed_feature_count] = action
        features[-self.user_feature_count:] = self.user_features

        candidate_features = []
        for f in available_actions:
            candidate_feature = np.copy(features)
            candidate_feature[
                self.feed_counts * self.feed_feature_count:(self.feed_counts + 1) * self.feed_feature_count
            ] = f
            candidate_features.append(candidate_feature)
        candidate_features = np.array(candidate_features)

        self.training_data.push(
            torch.tensor([self.latest_feature], dtype=torch.double).to(device),
            torch.tensor([reward], dtype=torch.double).to(device),
            torch.tensor([candidate_features], dtype=torch.double).to(device),
        )

    def learn_from_buffer(self):
        if len(self.training_data) < self.batch_size:
            return

        loss_ensemble = 0.
        for i in range(0, 10):
            transitions = self.training_data.sample(self.batch_size)
            batch = Transition(*zip(*transitions))
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                        batch.next_state)), device=device, dtype=torch.bool)
            state_batch = torch.cat(batch.state)
            reward_batch = torch.cat(batch.reward)
            state_action_values = self.model(state_batch)

            all_none = True
            for s in batch.next_state:
                if s is not None:
                    all_none = False

            next_state_values = torch.zeros(self.batch_size, device=device, dtype=torch.double)
            if not all_none:
                non_final_next_states = torch.cat([s for s in batch.next_state
                                                            if s is not None])

                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].reshape((-1)).detach()

            expected_state_action_values = self.gamma * next_state_values + reward_batch

            loss = self.loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))
            loss_ensemble += loss.item()

            self.optimizer.zero_grad()
            loss.backward()

            for param in self.model.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

        self.running_loss = 0.8 * self.running_loss + 0.2 * loss_ensemble
        self.epsilon = 0.999 * self.epsilon


    def reset(self, user_features, initial_feeds, user_embedding):
        self.cum_rewards: float = 0.
        self.interest_level = 0.
        self.latest_feature = None
        self.current_feed_candidates = initial_feeds
        self.target_net.load_state_dict(self.model.state_dict())
        self.target_net.double()
        self.target_net.eval()
        self.target_net.to(device)
        self.history_actions = []
        self.cum_reward_history.append(self.cum_rewards)
        self.current_feed = 0
        self.user_features = user_features
