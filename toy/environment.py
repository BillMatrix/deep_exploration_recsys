from helper import sigmoid
from uuid import UUID
import numpy as np
from typing import List, Set


class FeedUnit(object):
    def __init__(self, interest: int, unique_id: int):
        self.interest: int = interest
        self.index: int = unique_id


class Feed(object):
    def __init__(self, feeds: List[FeedUnit], num_positive):
        self.feeds = feeds
        self.interest_level: float = 0.
        self.seen_units = []
        self.current_feed = 0
        self.num_positive = num_positive

    def reset(self):
        self.interest_level = 0.
        self.current_feed = 0
        self.seen_units = []

    def step(self, action: int):
        reward = 0
        if action == 1:
            self.interest_level += self.feeds[self.current_feed].interest
#             if self.interest_level > self.num_positive:
#                 self.interest_level = self.num_positive

        if self.interest_level == self.num_positive:
            reward = 1

        if self.current_feed == len(self.feeds) or reward == 1:
            return False, reward, self.interest_level

        scroll: bool = True
        self.current_feed += 1

        if action == 1:
#             print('here')
#             reward -= 0.01/20

            scroll = self.interest_level >= 0 and self.current_feed != len(self.feeds)

            return scroll, reward, self.interest_level

        scroll = self.current_feed != len(self.feeds)
        return scroll, reward, self.interest_level
