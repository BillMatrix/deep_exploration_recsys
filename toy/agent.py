from environment import FeedUnit
from typing import List


class Agent(object):
    def __init__(self, feed_units: List[FeedUnit]):
        pass

    def choose_action(self, session_length: int):
        pass

    def update_buffer(self, scroll, num_ads, num_actions):
        pass

    def learn_from_buffer(self):
        pass