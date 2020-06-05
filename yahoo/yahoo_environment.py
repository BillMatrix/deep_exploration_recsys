from helper import sigmoid
from uuid import UUID
import numpy as np
from typing import List, Set


class YahooFeedUnit(object):
    def __init__(self, features, embedding, unique_id):
        self.features = features
        self.embedding = embedding
        self.index = unique_id


class YahooFeed(object):
    def __init__(self, feeds, target_interest_level, user_model, user_features, env_type='sparse_reward'):
        self.feeds = feeds
        self.interest_level = 0.
        self.seen_units = []
        self.current_feed = 0
        self.user_model = user_model
        self.user_features = user_features
        self.env_type = env_type

        self.target_interest_level = 0
        self.mean_interest = 0
        all_interest = []
        for feed_candidates in feeds:
            candidate_features = [np.concatenate((np.array(self.user_features), candidate.features)) for candidate in feed_candidates]
            # self.mean_interest += np.sum(
            #     self.user_model.predict_proba(candidate_features[:-1])[:, 1],
            # )
            all_interest.append(self.user_model.predict_proba(candidate_features[:-1])[:, 1])

        self.threshold = np.percentile(all_interest, 80)

        for feed_candidates in feeds:
            candidate_features = [np.concatenate((np.array(self.user_features), candidate.features)) for candidate in feed_candidates]
            self.target_interest_level += max(
                np.max(self.user_model.predict_proba(candidate_features[:-1])[:, 1]) - self.threshold,
                0.,
            )
        self.target_interest_level /= 1.25
        # print('target interest level: {}'.format(self.target_interest_level))

    def reset(self, user_features):
        self.interest_level = 0.
        self.current_feed = 0
        self.seen_units = []
        self.user_features = user_features

    def step(self, action_feed):
        reward = 0
        # print('start')
        # print('target level', self.target_interest_level)
        # print('features', action_feed.features)
        # print('no action', np.all(np.equal(action_feed.features, np.array([0. for _ in range(6)]))))

        if not np.all(np.equal(action_feed.features, np.array([0. for _ in range(6)]))):
            self.interest_level += self.user_model.predict_proba(
                [np.concatenate((np.array(self.user_features), action_feed.features))]
            )[0][1] - self.threshold

        if self.env_type == 'sparse_reward':
            if self.interest_level >= self.target_interest_level:
                reward = 1
        elif self.env_type == 'immediate_reward':
            if not np.all(np.equal(action_feed.features, np.array([0. for _ in range(6)]))):
                reward = float(self.user_model.predict_proba(
                    [np.concatenate((np.array(self.user_features), action_feed.features))]
                )[0][1] > self.threshold)

        # print(self.current_feed)
        # print('reward', reward)

        self.current_feed += 1

        scroll = self.interest_level >= 0 and self.current_feed != len(self.feeds)
        # print('current index', self.current_feed)
        # print('interest level', self.interest_level)
        # print('scroll', scroll)

        return scroll, reward, self.feeds[self.current_feed] if scroll else None
