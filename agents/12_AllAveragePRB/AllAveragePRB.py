import numpy as np
import random
from math import sqrt

from anyrl.rollouts import PrioritizedReplayBuffer


class AllAveragePRB(PrioritizedReplayBuffer):
    """
    A prioritized replay buffer with All Average Threshold caching and
    loss-proportional sampling.
    """

    def __init__(self, capacity, alpha, beta, first_max=1, epsilon=0):
        super().__init__(capacity, alpha, beta, first_max, epsilon)

        self.error_threshold = 0 # Threshold error for newly added sample

    def add_sample(self, sample, init_weight=None):
        """
        Add a sample to the buffer.
        When new samples are added without an explicit initial weight, the
        maximum weight argument ever seen is used. When the buffer is empty,
        first_max is used.
        """
        if init_weight is None:
            new_error = self._process_weight(self._max_weight_arg)

            self.transitions.append(sample)
            self.errors.append(new_error)

            # Update error_threshold via moving average
            self.error_threshold += (new_error - self.error_threshold) / len(self.transitions)
        else:
            new_error = self._process_weight(init_weight)

            if new_error >= self.error_threshold:
                self.transitions.append(sample)
                self.errors.append(new_error)

            # Update error_threshold via moving average
            self.error_threshold += (new_error - self.error_threshold) / len(self.transitions)

        while len(self.transitions) > self.capacity:
            del self.transitions[0]
