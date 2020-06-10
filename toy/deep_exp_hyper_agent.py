from environment import FeedUnit
from typing import List, Dict, Tuple
from agent import Agent
import torch
import itertools
import torch.optim as optim
import torch.nn.functional as f
import torch.nn as nn
import math
import copy
import numpy as np
from helper import DQNWithPrior, ReplayMemory, Transition, MLP, HyperDQN, HyperDQNWithPrior

has_gpu = torch.cuda.is_available()
device = torch.device("cpu")


class DeepExpHyperAgent(Agent):
    def __init__(
        self,
        feed_units: List[int],
        agent_name: str,
        index_size: int = 1,
        prior_variance: float = 1.0,
        model_dims: List[int] = [],
        lr: float = 1e-4,
        batch_size: int = 1,
        noise_variance = 0
    ):
        self.feed_units = copy.deepcopy(feed_units)
#         self.available_units = copy.deepcopy(feed_units)
        self.agent_name = agent_name

        self.cum_rewards: float = 0.
        self.interest_level = 0.
        self.num_features: int = len(feed_units) + 1
        self.noise_variance = noise_variance

        self.index_size: int = index_size
        self.training_data = ReplayMemory(100000)

        self.latest_feature = None
        self.latest_action = None

        self.prior_variance = prior_variance

        self.model_dims: List[int] = [self.num_features] + model_dims + [2]
        self.priors = []
        for i in range(self.index_size):
            self.priors.append(MLP(self.model_dims))
            self.priors[i].initialize()
            self.priors[i].double()
            self.priors[i].eval()
            self.priors[i].to(device)
        self.embed = nn.Embedding(self.index_size, self.index_size)

        self.cur_index = np.random.choice(self.index_size)
        self.cur_z = f.one_hot(torch.LongTensor([self.cur_index]), self.index_size)
        self.cur_z = self.cur_z.double()

        self.hyper_model = HyperDQNWithPrior(
            [index_size, 15],
            self.model_dims,
            index_size,
            self.priors[self.cur_index],
            scale=np.sqrt(self.prior_variance)
        )
        self.hyper_model.double()

        self.target_net = HyperDQNWithPrior(
            [index_size, 15],
            self.model_dims,
            index_size,
            self.priors[self.cur_index],
            np.sqrt(self.prior_variance)
        )
        self.target_net.load_state_dict(self.hyper_model.state_dict())
        self.target_net.double()
        self.target_net.eval()

        self.loss_fn = torch.nn.MSELoss(reduction='sum')

        self.optimizer = optim.Adam(self.hyper_model.parameters(), lr=lr)
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
            outcomes = self.target_net(
                torch.tensor([features], dtype=torch.double),
                self.embed(torch.tensor([self.cur_index])).double(),
            )
            # print(outcomes)

            _, best_index = torch.max(outcomes, 1)
            # print(best_index)
            best_index = best_index.item()

            if np.random.rand() < 0.05:
                best_index = np.random.choice(len(outcomes.numpy().tolist()))

            best_action = [available_actions[best_index]]
            self.latest_feature = features
            self.latest_action = best_action
            if best_action[0] == 1:
                self.history_unit_indices.append(self.current_feed)

            self.current_feed += 1

            # print('action: {}'.format(best_action[0]))
            return best_action[0]

    def update_buffer(
        self,
        scroll: bool,
        reward: int,
    ):
        # print('reward: {}'.format(reward))
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
        # try:
        for _ in range(5):
            transitions = self.training_data.sample(self.batch_size)
            if (len(transitions) < self.batch_size):
                print('insufficient')
            batch = Transition(*zip(*transitions))

            all_none = True
            for s in batch.next_state:
                if s is not None:
                    all_none = False

            if not all_none:
                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool).expand([
                                                self.index_size, -1
                                            ]).reshape((self.index_size * self.batch_size)) # [index_size * batch_size]
                non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).expand([
                    self.index_size, -1, -1
                ]) # [index_size, -1, feature_size]

            state_batch = torch.cat(batch.state).expand([self.index_size, -1, -1]).reshape(
                [self.index_size * self.batch_size, -1]) # [index_size * batch_size, feature_size]
            action_batch = torch.cat(batch.action).expand([self.index_size, -1, -1]).reshape(
                [self.index_size * self.batch_size, -1]
            ) # [index_size * batch_size, action_size]
            reward_batch = torch.cat(batch.reward).expand([self.index_size, -1]).reshape(
                [self.index_size * self.batch_size]
            )  # [index_size * batch_size]
            z_batch = torch.tensor([
                self.embed(torch.LongTensor([j])).expand([self.batch_size, -1]).detach().numpy()
                for j in range(self.index_size)
            ]).view(
                [self.index_size * self.batch_size, -1]
            ) # [index_size * batch_size, index_size]
            z_batch = z_batch.double()
            # print(state_batch.shape)
            # print(z_batch.shape)
            state_action_values = self.hyper_model(state_batch, z_batch).gather(1, action_batch)

            next_state_values = torch.zeros(self.batch_size * self.index_size, device=device, dtype=torch.double)

            if not all_none:
                z_batch_for_next_states = torch.tensor([
                    self.embed(torch.LongTensor([j])).expand([non_final_next_states.shape[1], -1]).detach().numpy()
                    for j in range(self.index_size)
                ]).view(
                    [self.index_size * non_final_next_states.shape[1], -1]
                ).double()
                non_final_next_states = non_final_next_states.reshape(
                    [self.index_size * non_final_next_states.shape[1], -1]) # [index_size * -1, feature_size]
                # print('target output shape: ', self.target_net(non_final_next_states, z_batch_for_next_states).shape)
                next_state_values[non_final_mask] = self.target_net(non_final_next_states, z_batch_for_next_states).max(1)[0].detach()

            expected_state_action_values = self.gamma * next_state_values + reward_batch

            loss = self.loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))
            l2_norm = 0
            for param in self.hyper_model.parameters():
                l2_norm += torch.norm(param)
            loss += 0.5 * l2_norm
            loss_ensemble += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            # print(self.hyper_model.f_diff.bias_hyper_networks[0].layers[0].weight.grad.data)

            for param in self.hyper_model.parameters():
                param.grad.data.clamp_(-1, 1)
            # print(self.optimizer.state_dict())
            self.optimizer.step()

            self.running_loss = 0.8 * self.running_loss + 0.2 * loss_ensemble
        # except:
        #     print('{}: no non-terminal state'.format(self.agent_name))

    def reset(self):
        self.available_units = copy.deepcopy(self.feed_units)
        self.cum_rewards: float = 0.
        self.interest_level = 0.
        self.latest_feature = None
        self.latest_action = None
#         self.history_unit_indices = []
        self.cum_reward_history.append(self.cum_rewards)

        self.target_net.load_state_dict(self.hyper_model.state_dict())
        self.target_net.double()
        self.target_net.eval()

        self.cur_index = np.random.choice(self.index_size)
        self.cur_z = f.one_hot(torch.LongTensor([self.cur_index]), self.index_size)
        self.cur_z = self.cur_z.double()
        self.hyper_model.set_prior(self.priors[self.cur_index])
        self.target_net.set_prior(self.priors[self.cur_index])

        self.history_unit_indices = []
        self.cum_reward_history.append(self.cum_rewards)
        self.current_feed = 0
        self.current_loc = [0, 0]
