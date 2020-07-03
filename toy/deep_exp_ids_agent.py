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


class DeepExpIDSAgent(Agent):
    def __init__(
        self,
        feed_units: List[int],
        agent_name: str,
        ensemble_size: int = 100,
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
            all_outcomes = [self.target_nets[model_index](
                torch.tensor(features, dtype=torch.double)
            ) for model_index in range(self.ensemble_size)]

            mean_immediate_regret = self.mean_immediate_regret(all_outcomes)
            var_immediate_regret = self.var_immediate_regret(all_outcomes, len(available_actions))
            best_index = self.best_ids_action(mean_immediate_regret, var_immediate_regret)

            best_action = [available_actions[best_index]]
            self.latest_feature = features
            self.latest_action = best_action
            if best_action[0] == 1:
                self.history_unit_indices.append(self.current_feed)

            self.current_feed += 1

            # print('action: {}'.format(best_action[0]))
            return best_action[0]

    def mean_immediate_regret(self, all_outcomes):
        sum_immediate_regret = None
        for model_index in range(self.ensemble_size):
            outcomes = all_outcomes[model_index]
            max_outcome, _ = torch.max(outcomes, 0)
            if sum_immediate_regret is None:
                sum_immediate_regret = max_outcome - outcomes
            else:
                sum_immediate_regret += max_outcome - outcomes

        return sum_immediate_regret / self.ensemble_size

    def var_immediate_regret(self, all_outcomes, num_actions):
        count_best_outcome = [0 for _ in range(num_actions)]
        sum_out_best = {}
        sum_out_all = None

        for model_index in range(self.ensemble_size):
            outcomes = all_outcomes[model_index]
            max_outcome, best_index = torch.max(outcomes, 0)
            count_best_outcome[best_index] += 1
            if best_index in sum_out_best:
                sum_out_best[best_index] += outcomes
            else:
                sum_out_best[best_index] = outcomes

            if sum_out_all is None:
                sum_out_all = outcomes
            else:
                sum_out_all += outcomes

        var = torch.tensor([0. for _ in range(num_actions)]).double()
        for a in range(num_actions):
            if a not in sum_out_best:
                sum_out_best[a] = torch.tensor(
                    [0. for _ in range(num_actions)]
                ).double()

            coeff = count_best_outcome[a] / self.ensemble_size
            if coeff == 0:
                continue

            sum_err = (
                1 / count_best_outcome[a] * sum_out_best[a][a]
                - 1 / num_actions * sum_out_all[a]
            ) ** 2
            var[a] = coeff * sum_err.item()

        return var

    def best_ids_action(self, mean_immediate_regret, var_immediate_regret):
        regret_sq = mean_immediate_regret ** 2
        info_gain = torch.log(1 + var_immediate_regret) + 1e-5
        return torch.argmin(regret_sq / info_gain)

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
        # except:
            # print('{}: no non-terminal state'.format(self.agent_name))

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
