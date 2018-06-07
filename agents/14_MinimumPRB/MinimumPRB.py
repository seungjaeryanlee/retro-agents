import numpy as np
import random
from math import sqrt

from anyrl.rollouts import PrioritizedReplayBuffer


class MinimumPRB(PrioritizedReplayBuffer):
    """
    A prioritized replay buffer with Minimum Threshold caching and
    loss-proportional sampling.
    """

    def add_sample(self, sample, init_weight=None):
        """
        Add a sample to the buffer.
        When new samples are added without an explicit initial weight, the
        maximum weight argument ever seen is used. When the buffer is empty,
        first_max is used.
        """
        if init_weight is None:
            new_error = self._process_weight(self._max_weight_arg)
        else:
            new_error = self._process_weight(init_weight)

        if new_error < self.errors.min():
            return

        self.transitions.append(sample)
        self.errors.append(new_error)

        while len(self.transitions) > self.capacity:
            del self.transitions[0]
