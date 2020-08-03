from yahoo_environment import YahooFeedUnit
from typing import List, Dict, Tuple
import torch
import itertools
import torch.optim as optim
import math
import copy
import numpy as np
from helper import NCFWithPrior, NCFReplayMemory, NCFReplayTransition, NCF, MLP

has_gpu = torch.cuda.is_available()
device = torch.device("cuda" if has_gpu else "cpu")


class YahooDeepExpNCFAgent():
    def __init__(
        self,
        initial_feed_candidates,
        user_features,
        user_embedding,
        feed_counts,
        agent_name: str,
        feed_feature_count = 6,
        feed_num = 50,
        user_num = 20,
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
        self.user_embedding = user_embedding
        self.feed_counts = feed_counts
        self.agent_name = agent_name
        self.feed_num = feed_num
        self.user_num = user_num

        self.cum_rewards: float = 0.
        self.interest_level = 0.
        self.feed_feature_count = feed_feature_count
        self.user_feature_count = user_feature_count
        self.num_features = feed_counts * feed_feature_count + feed_feature_count + user_feature_count
        self.noise_variance = noise_variance

        self.ensemble_size: int = ensemble_size
        self.training_datas = [NCFReplayMemory(100000) for _ in range(self.ensemble_size)]

        self.latest_feature = None
        self.latest_feed_embedding = None
        self.latest_user_embedding = None

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
            cur_model = NCF(
                user_num=self.user_num,
                item_num=self.feed_num,
                dense_feature_size=self.num_features,
                factor_num=32,
                num_layers=3,
                dropout=0.0,
                model='NeuMF-end',
            ).double()
            self.models.append(NCFWithPrior(cur_model, priors[i], scale=np.sqrt(self.prior_variance)))
            self.models[i].double()
            self.models[i].to(device)

        self.target_nets: List[DQNWithPrior] = []
        for i in range(self.ensemble_size):
            cur_model = NCF(
                user_num=self.user_num,
                item_num=self.feed_num,
                dense_feature_size=self.num_features,
                factor_num=32,
                num_layers=3,
                dropout=0.0,
                model='NeuMF-end',
            ).double()
            self.target_nets.append(NCFWithPrior(cur_model, priors[i], scale=np.sqrt(self.prior_variance)))
            self.target_nets[i].load_state_dict(self.models[i].state_dict())
            self.target_nets[i].double()
            self.target_nets[i].eval()
            self.target_nets[i].to(device)

        self.loss_fn = torch.nn.MSELoss(reduction='sum')

        self.optimizers = []
        for i in range(self.ensemble_size):
            self.optimizers.append(optim.Adam(self.models[i].parameters(), lr=lr))

        self.cur_index = np.random.choice(self.ensemble_size)
        self.cur_net = self.target_nets[self.cur_index]
        self.batch_size = batch_size
        self.gamma = 0.99
        self.running_loss = 0.0
        self.history_actions = []
        self.cum_reward_history: List[float] = []
        self.current_feed = 0

    def choose_action(self):
        available_actions = [candidate for candidate in self.current_feed_candidates]

        features = [-1. for _ in range(self.num_features)]
        for index, action in enumerate(self.history_actions):
            features[index * self.feed_feature_count:(index + 1) * self.feed_feature_count] = action
        features[-self.user_feature_count:] = self.user_features

        candidate_features = []
        candidate_user_embeddings = []
        candidate_feed_embeddings = []
        for f in available_actions:
            candidate_user_embeddings.append(self.user_embedding)
            candidate_feed_embeddings.append(f.embedding)
            candidate_feature = np.copy(features)
            candidate_feature[
                self.feed_counts * self.feed_feature_count:(self.feed_counts + 1) * self.feed_feature_count
            ] = f.features
            candidate_features.append(candidate_feature)
        candidate_features = np.array(candidate_features)

#         base_feature.append(self.interest_level)
        with torch.no_grad():
            outcomes = self.cur_net(
                torch.tensor(candidate_user_embeddings, dtype=torch.long).to(device),
                torch.tensor(candidate_feed_embeddings, dtype=torch.long).to(device),
                torch.tensor(candidate_features, dtype=torch.double).to(device)
            )

            _, best_index = torch.max(outcomes, 0)
            best_index = best_index.item()

            best_action = self.current_feed_candidates[best_index]
            self.latest_feature = candidate_features[best_index]
            self.latest_user_embedding = candidate_user_embeddings[best_index]
            self.latest_feed_embedding = candidate_feed_embeddings[best_index]
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
            self.training_datas[self.cur_index].push(
                torch.tensor([self.latest_feature], dtype=torch.double).to(device),
                torch.tensor([self.latest_user_embedding], dtype=torch.long).to(device),
                torch.tensor([self.latest_feed_embedding], dtype=torch.long).to(device),
                torch.tensor([reward], dtype=torch.double).to(device),
                None,
                None,
                None,
            )
            return

        available_actions = [candidate for candidate in self.current_feed_candidates]
        features: List[float] = [-1. for _ in range(self.num_features)]
        for index, action in enumerate(self.history_actions):
            features[index * self.feed_feature_count:(index + 1) * self.feed_feature_count] = action
        features[-self.user_feature_count:] = self.user_features

        candidate_features = []
        candidate_user_embeddings = []
        candidate_feed_embeddings = []
        for f in available_actions:
            candidate_feature = np.copy(features)
            candidate_feed_embeddings.append(f.embedding)
            candidate_user_embeddings.append(self.user_embedding)
            candidate_feature[
                self.feed_counts * self.feed_feature_count:(self.feed_counts + 1) * self.feed_feature_count
            ] = f.features
            candidate_features.append(candidate_feature)
        candidate_features = np.array(candidate_features)

        self.training_datas[self.cur_index].push(
            torch.tensor([self.latest_feature], dtype=torch.double).to(device),
            torch.tensor([self.latest_user_embedding], dtype=torch.long).to(device),
            torch.tensor([self.latest_feed_embedding], dtype=torch.long).to(device),
            torch.tensor([reward], dtype=torch.double).to(device),
            torch.tensor([candidate_features], dtype=torch.double).to(device),
            torch.tensor([candidate_user_embeddings], dtype=torch.long).to(device),
            torch.tensor([candidate_feed_embeddings], dtype=torch.long).to(device),
        )

    def learn_from_buffer(self):
        for i in range(self.ensemble_size):
            if len(self.training_datas[i]) < self.batch_size:
                return

        loss_ensemble = 0.0
        for _ in range(10):
            for i in range(self.ensemble_size):
                transitions = self.training_datas[i].sample(self.batch_size)
                batch = NCFReplayTransition(*zip(*transitions))
                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                        batch.next_state)), device=device, dtype=torch.bool)
                state_batch = torch.cat(batch.state)
                user_embedding_batch = torch.cat(batch.user_embedding)
                feed_embedding_batch = torch.cat(batch.cur_feed_embedding)
                reward_batch = torch.cat(batch.reward)
                state_action_values = self.models[i](user_embedding_batch, feed_embedding_batch, state_batch)

                all_none = True
                for s in batch.next_state:
                    if s is not None:
                        all_none = False

                next_state_values = torch.zeros(self.batch_size, device=device, dtype=torch.double)
                if not all_none:
                    non_final_next_states = torch.cat([s for s in batch.next_state
                                                                if s is not None])
                    non_final_feed_embeddings = torch.cat([s for s in batch.next_feed_embeddings if s is not None])
                    non_final_user_embeddings = torch.cat([s for s in batch.next_user_embeddings if s is not None])

                    next_state_values[non_final_mask] = self.target_nets[i](
                        non_final_user_embeddings,
                        non_final_feed_embeddings,
                        non_final_next_states,
                        2,
                    ).max(1)[0].reshape((-1)).detach()

                expected_state_action_values = self.gamma * next_state_values + reward_batch

                loss = self.loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))
                loss_ensemble += loss.item()

                self.optimizers[i].zero_grad()
                loss.backward()

                for param in self.models[i].parameters():
                    param.grad.data.clamp_(-1, 1)
                self.optimizers[i].step()

            self.running_loss = 0.8 * self.running_loss + 0.2 * loss_ensemble

    def reset(self, user_features, initial_feeds, user_embedding):
        self.cum_rewards: float = 0.
        self.interest_level = 0.
        self.latest_feature = None
        self.current_feed_candidates = initial_feeds
#         self.history_unit_indices = []
        self.cum_reward_history.append(self.cum_rewards)

        for i in range(self.ensemble_size):
            self.target_nets[i].load_state_dict(self.models[i].state_dict())
            self.target_nets[i].double()
            self.target_nets[i].eval()
            self.target_nets[i].to(device)

        self.cur_index = np.random.choice(self.ensemble_size)
        self.cur_net = self.target_nets[self.cur_index]
        self.history_actions = []
        self.cum_reward_history.append(self.cum_rewards)
        self.current_feed = 0
        self.user_features = user_features
        self.user_embedding = user_embedding
