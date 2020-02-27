from environment import FeedUnit
from typing import List, Dict, Tuple
from agent import Agent
import torch
import itertools
import torch.optim as optim
import math
import copy
import numpy as np
from helper import DQNWithPrior, ReplayMemory, Transition, MLP

has_gpu = torch.cuda.is_available()
device = torch.device("cpu")


class DeepExpAgent(Agent):
    def __init__(
        self,
        feed_units: List[int],
        agent_name: str,
        ensemble_size: int = 10,
        prior_variance: float = 1.0,
        model_dims: List[int] = [20],
        lr: float = 1e-3,
        batch_size: int = 128,
        noise_variance = 0
    ):
        self.feed_units = copy.deepcopy(feed_units)
#         self.available_units = copy.deepcopy(feed_units)
        self.agent_name = agent_name

        self.cum_rewards: float = 0.
        self.interest_level = 0.
        self.num_features: int = len(feed_units) + 1
        self.noise_variance = noise_variance

        self.ensemble_size: int = ensemble_size
        self.training_data = ReplayMemory(100000)
#         self.training_datas = []
#         for i in range(self.ensemble_size):
#             self.training_datas.append(ReplayMemory(100000))

        self.latest_feature = None
        self.latest_action = None

        self.prior_variance = prior_variance

        self.model_dims: List[int] = [self.num_features] + model_dims + [2]
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
        self.history_unit_indices: List[int] = []
        self.cum_reward_history: List[float] = []
        self.current_feed = 0

    def choose_action(self):
        available_actions = [0, 1]

        features: List[float] = [-1. for _ in range(self.num_features)]
        for index in range(self.current_feed):
            features[index] = 0.
        for index in self.history_unit_indices:
            features[index] = 1.

        with torch.no_grad():
            outcomes = self.cur_net(
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

            return best_action[0]

    def update_buffer(
        self,
        scroll: bool,
        reward: int,
        interest = 0
    ):
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

        loss_ensemble = 0.0
        try:
            for _ in range(10):
                transitions = self.training_data.sample(self.batch_size)
                for i in range(self.ensemble_size):
                    batch = Transition(*zip(*transitions))
                    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.bool)
                    non_final_next_states = torch.cat([s for s in batch.next_state
                                                                if s is not None])

                    state_batch = torch.cat(batch.state)
                    action_batch = torch.cat(batch.action)
                    reward_batch = torch.cat(batch.reward)
                    state_action_values = self.models[i](state_batch).gather(1, action_batch)

                    next_state_values = torch.zeros(self.batch_size, device=device, dtype=torch.double)
                    next_state_values[non_final_mask] = self.target_nets[i](non_final_next_states).max(1)[0].detach()

                    expected_state_action_values = self.gamma * next_state_values + reward_batch

                    loss = self.loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))
                    loss_ensemble += loss.item()

                    self.optimizers[i].zero_grad()
                    loss.backward()

        #             for param in self.model.parameters():
        #                 param.grad.data.clamp_(-1, 1)
                    self.optimizers[i].step()

                self.running_loss = 0.8 * self.running_loss + 0.2 * loss_ensemble
        except:
            print('no non-terminal state')

    def reset(self):
        self.available_units = copy.deepcopy(self.feed_units)
        self.cum_rewards: float = 0.
        self.interest_level = 0.
        self.latest_feature = None
        self.latest_action = None
#         self.history_unit_indices = []
        self.cum_reward_history.append(self.cum_rewards)

        for i in range(self.ensemble_size):
            self.target_nets[i].load_state_dict(self.models[i].state_dict())
            self.target_nets[i].double()
            self.target_nets[i].eval()
            self.target_nets[i].to(device)

        self.cur_net = self.target_nets[np.random.choice(self.ensemble_size)]
        self.history_unit_indices = []
        self.cum_reward_history.append(self.cum_rewards)
        self.current_feed = 0
        self.current_loc = [0, 0]
