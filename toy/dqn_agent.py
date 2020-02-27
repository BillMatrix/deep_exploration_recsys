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
from helper import MLP, Transition, ReplayMemory

has_gpu = torch.cuda.is_available()
device = torch.device("cpu")


class DQNAgent(Agent):
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
        self.training_data: ReplayMemory = ReplayMemory(100000)

        self.model_dims: List[int] = [self.num_features] + model_dims + [2]
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
        self.history_unit_indices: List[int] = []
        self.latest_feature = None
        self.latest_action = None
        self.current_feed = 0
        self.cum_reward_history: List[float] = []
        self.current_loc = [0, 0]

    def choose_action(self):
        available_actions = [0, 1]

        features: List[float] = [-1. for _ in range(self.num_features)]
        for index in range(self.current_feed):
            features[index] = 0.
        for index in self.history_unit_indices:
            features[index] = 1.

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
        interest = 0
    ):
#         print(reward)
        self.cum_rewards += reward
        if not scroll:
            self.training_data.push(
                torch.tensor([self.latest_feature], dtype=torch.double),
                torch.tensor([self.latest_action], dtype=torch.long),
                torch.tensor([reward], dtype=torch.double),
                None,
            )
            return

        features: List[float] = [-1. for _ in range(self.num_features)]
        for index in range(self.current_feed):
            features[index] = 0.
        for index in self.history_unit_indices:
            features[index] = 1.

        self.training_data.push(
            torch.tensor([self.latest_feature], dtype=torch.double),
            torch.tensor([self.latest_action], dtype=torch.long),
            torch.tensor([reward], dtype=torch.double),
            torch.tensor([features], dtype=torch.double),
        )

    def learn_from_buffer(self):
        if len(self.training_data) < self.batch_size:
            return

        try:
            loss_ensemble = 0.
            for i in range(0, 10):
                transitions = self.training_data.sample(self.batch_size)
                batch = Transition(*zip(*transitions))
                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
                non_final_next_states = torch.cat([s for s in batch.next_state
                                                            if s is not None])

                state_batch = torch.cat(batch.state)
                action_batch = torch.cat(batch.action)
                reward_batch = torch.cat(batch.reward)
                state_action_values = self.model(state_batch).gather(1, action_batch)

                next_state_values = torch.zeros(self.batch_size, device=device, dtype=torch.double)
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

                expected_state_action_values = self.gamma * next_state_values + reward_batch

                loss = self.loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))
                loss_ensemble += loss.item()

                self.optimizer.zero_grad()
                loss.backward()

    #             for param in self.model.parameters():
    #                 param.grad.data.clamp_(-1, 1)
                self.optimizer.step()

            self.running_loss = 0.8 * self.running_loss + 0.2 * loss_ensemble
            self.epsilon = 0.999 * self.epsilon
        except:
            print('no non-terminal state')

    def reset(self):
        self.cum_rewards: float = 0.
        self.interest_level = 0.
        self.latest_feature = None
        self.latest_action = None
        self.target_net.load_state_dict(self.model.state_dict())
        self.target_net.double()
        self.target_net.eval()
        self.target_net.to(device)
        self.history_unit_indices = []
        self.cum_reward_history.append(self.cum_rewards)
        self.current_loc = [0, 0]
        self.current_feed = 0
