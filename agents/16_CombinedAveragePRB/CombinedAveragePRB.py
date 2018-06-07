import numpy as np
import random
from math import sqrt

from anyrl.rollouts import PrioritizedReplayBuffer

class CombinedAveragePRB(PrioritizedReplayBuffer):
    """
    A prioritized replay buffer with threshold of a weighted average of all
    average and buffer average and loss-proportional sampling.
    """

    def __init__(self, capacity, alpha, beta, first_max=1, epsilon=0, gamma=0.5):
        super().__init__(capacity, alpha, beta, first_max, epsilon)

        self.gamma = gamma
        self.buffer_average = 0
        self.all_average = 0

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

        if new_error >= self.gamma * self.buffer_average + (1 - self.gamma) *self.all_average:
            self.transitions.append(sample)
            self.errors.append(new_error)

            self.buffer_average += (new_error - self.buffer_average) / len(self.transitions)

        self.all_average += (new_error - self.all_average) / len(self.transitions)

        while len(self.transitions) > self.capacity:
            del self.transitions[0]
