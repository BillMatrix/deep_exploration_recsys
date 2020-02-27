from yahoo_environment import YahooFeedUnit
from typing import List, Dict, Tuple
import torch
import itertools
import torch.optim as optim
import math
import copy
import numpy as np
from helper import DQNWithPrior, ReplayMemory, Transition, MLP

has_gpu = torch.cuda.is_available()
device = torch.device("cpu")


class YahooDeepExpAgent():
    def __init__(
        self,
        initial_feed_candidates,
        user_features,
        feed_counts,
        agent_name: str,
        feed_feature_count = 6,
        user_feature_count = 6,
        ensemble_size: int = 10,
        prior_variance: float = 1.0,
        model_dims: List[int] = [50, 25],
        lr: float = 1e-3,
        batch_size: int = 128,
        noise_variance = 0
    ):
        self.initial_feed_candidates = initial_feed_candidates
        self.current_feed_candidates = initial_feed_candidates
        self.user_features = user_features
        self.feed_counts = feed_counts
        self.agent_name = agent_name

        self.cum_rewards: float = 0.
        self.interest_level = 0.
        self.feed_feature_count = feed_feature_count
        self.user_feature_count = user_feature_count
        self.num_features = feed_counts * feed_feature_count + feed_feature_count + user_feature_count
        self.noise_variance = noise_variance

        self.ensemble_size: int = ensemble_size
        self.training_data = ReplayMemory(100000)

        self.latest_feature = None

        self.prior_variance = prior_variance

        self.model_dims: List[int] = [self.num_features] + model_dims + [1]
        priors = []

        for i in range(self.ensemble_size):
            priors.append(MLP(self.model_dims))
            priors[i].initialize()
            priors[i].double()
            priors[i].eval()
            priors[i].to(device)

        self.models: List[DQNWithPrior] = []
        for i in range(self.ensemble_size):
            self.models.append(DQNWithPrior(self.model_dims, priors[i], scale=np.sqrt(self.prior_variance)))
            self.models[i].initialize()
            self.models[i].double()
            self.models[i].to(device)

        self.target_nets: List[DQNWithPrior] = []
        for i in range(self.ensemble_size):
            self.target_nets.append(DQNWithPrior(self.model_dims, priors[i], scale=np.sqrt(self.prior_variance)))
            self.target_nets[i].load_state_dict(self.models[i].state_dict())
            self.target_nets[i].double()
            self.target_nets[i].eval()
            self.target_nets[i].to(device)

        self.loss_fn = torch.nn.MSELoss(reduction='sum')

        self.optimizers = []
        for i in range(self.ensemble_size):
            self.optimizers.append(optim.Adam(self.models[i].parameters(), lr=lr))

        self.cur_net = self.target_nets[np.random.choice(self.ensemble_size)]
        self.batch_size = batch_size
        self.gamma = 0.99
        self.running_loss = 0.0
        self.history_actions = []
        self.cum_reward_history: List[float] = []
        self.current_feed = 0

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
            outcomes = self.cur_net(
                torch.tensor(candidate_features, dtype=torch.double)
            )

            _, best_index = torch.max(outcomes, 0)
            best_index = best_index.item()

            best_action = self.current_feed_candidates[best_index]
            self.latest_feature =  candidate_features[best_index]
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
        self.current_feed_candidates = new_batch
        if not scroll:
            self.training_data.push(
                torch.tensor([self.latest_feature], dtype=torch.double),
                torch.tensor([reward], dtype=torch.double),
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
            torch.tensor([self.latest_feature], dtype=torch.double),
            torch.tensor([reward], dtype=torch.double),
            torch.tensor([candidate_features], dtype=torch.double),
        )

    def learn_from_buffer(self):
        if len(self.training_data) < self.batch_size:
            return

        loss_ensemble = 0.0
        for _ in range(10):
            transitions = self.training_data.sample(self.batch_size)
            for i in range(self.ensemble_size):
                batch = Transition(*zip(*transitions))
                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)

                state_batch = torch.cat(batch.state)
                reward_batch = torch.cat(batch.reward)
                state_action_values = self.models[i](state_batch)

                all_none = True
                for s in batch.next_state:
                    if s is not None:
                        all_none = False

                next_state_values = torch.zeros(self.batch_size, device=device, dtype=torch.double)
                if not all_none:
                    non_final_next_states = torch.cat([s for s in batch.next_state
                                                                if s is not None])

                    next_state_values[non_final_mask] = self.target_nets[i](non_final_next_states).max(1)[0].reshape((-1)).detach()

                expected_state_action_values = self.gamma * next_state_values + reward_batch

                loss = self.loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))
                loss_ensemble += loss.item()

                self.optimizers[i].zero_grad()
                loss.backward()

                for param in self.models[i].parameters():
                    param.grad.data.clamp_(-1, 1)
                self.optimizers[i].step()

            self.running_loss = 0.8 * self.running_loss + 0.2 * loss_ensemble

    def reset(self, user_features):
        self.cum_rewards: float = 0.
        self.interest_level = 0.
        self.latest_feature = None
#         self.history_unit_indices = []
        self.cum_reward_history.append(self.cum_rewards)

        for i in range(self.ensemble_size):
            self.target_nets[i].load_state_dict(self.models[i].state_dict())
            self.target_nets[i].double()
            self.target_nets[i].eval()
            self.target_nets[i].to(device)

        self.cur_net = self.target_nets[np.random.choice(self.ensemble_size)]
        self.history_actions = []
        self.cum_reward_history.append(self.cum_rewards)
        self.current_feed = 0
        self.current_feed_candidates = self.initial_feed_candidates
        self.user_features = user_features
