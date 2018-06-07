import numpy as np
import random
from math import sqrt

from anyrl.rollouts import PrioritizedReplayBuffer


class StochasticDeletionPRB(PrioritizedReplayBuffer):
    """
    A prioritized replay buffer with stochastic deletion and loss-proportional
    sampling.
    """

    def __init__(self, capacity, alpha, beta, first_max=1, epsilon=0):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.transitions = []
        self.errors = CustomFloatBuffer(capacity)
        self._max_weight_arg = first_max

    def add_sample(self, sample, init_weight=None):
        """
        Add a sample to the buffer.
        When new samples are added without an explicit initial weight, the
        maximum weight argument ever seen is used. When the buffer is empty,
        first_max is used.
        """
        self.transitions.append(sample)
        if init_weight is None:
            self.errors.append(self._process_weight(self._max_weight_arg))
        else:
            self.errors.append(self._process_weight(init_weight))
        while len(self.transitions) > self.capacity:
            del self.transitions[self.errors.inverse_sample(1)[0]]


class CustomFloatBuffer:
    """A ring-buffer of floating point values."""

    def __init__(self, capacity, dtype='float64'):
        self._capacity = capacity
        self._start = 0
        self._used = 0
        self._buffer = np.zeros((capacity,), dtype=dtype)
        self._bin_size = int(sqrt(capacity))
        num_bins = capacity // self._bin_size
        if num_bins * self._bin_size < capacity:
            num_bins += 1
        self._bin_sums = np.zeros((num_bins,), dtype=dtype)
        self._min = 0
        self._min_id = 0

    def append(self, value):
        """
        Add a value to the end of the buffer.
        If the buffer is full, the first value is removed.
        """
        idx = (self._start + self._used) % self._capacity
        if self._used < self._capacity:
            self._used += 1
        else:
            self._start = (self._start + 1) % self._capacity
        self._set_idx(idx, value)

    def sample(self, num_values):
        """
        Sample indices in proportion to their value.
        Returns:
          A tuple (indices, probs)
        """
        assert self._used >= num_values
        res = []
        probs = []
        bin_probs = self._bin_sums / np.sum(self._bin_sums)
        while len(res) < num_values:
            bin_idx = np.random.choice(len(self._bin_sums), p=bin_probs)
            bin_values = self._bin(bin_idx)
            sub_probs = bin_values / np.sum(bin_values)
            sub_idx = np.random.choice(len(bin_values), p=sub_probs)
            idx = bin_idx * self._bin_size + sub_idx
            res.append(idx)
            probs.append(bin_probs[bin_idx] * sub_probs[sub_idx])
        return (np.array(list(res)) - self._start) % self._capacity, np.array(probs)

    def inverse_sample(self, num_values):
        """
        Sample indices in inverse proportion to their value. The sampling used
        is $e^{-x}$.

        Returns:
          List of indices sampled.
        """
        assert self._used >= num_values
        res = []
        e_neg_bin_sums = np.exp(-1 * self._bin_sums)
        bin_probs = e_neg_bin_sums / np.sum(e_neg_bin_sums)
        while len(res) < num_values:
            bin_idx = np.random.choice(len(self._bin_sums), p=bin_probs)
            bin_values = self._bin(bin_idx)

            e_neg_bin_values = np.exp(-1 * bin_values)
            sub_probs = e_neg_bin_values / np.sum(e_neg_bin_values)
            sub_idx = np.random.choice(len(bin_values), p=sub_probs)
            idx = bin_idx * self._bin_size + sub_idx
            res.append(idx)
        return (np.array(list(res)) - self._start) % self._capacity

    def set_value(self, idx, value):
        """Set the value at the given index."""
        idx = (idx + self._start) % self._capacity
        self._set_idx(idx, value)

    def min(self):
        """Get the minimum value in the buffer."""
        return self._min
    
    def min_id(self):
        """Get the index of minimum value in the buffer."""
        return self._min_id

    def sum(self):
        """Get the sum of the values in the buffer."""
        return np.sum(self._bin_sums)

    def _set_idx(self, idx, value):
        assert not np.isnan(value)
        assert value > 0

        needs_recompute_min = False
        if self._min == self._buffer[idx]:
            needs_recompute_min = True
        elif value < self._min:
            self._min = value

        bin_idx = idx // self._bin_size
        self._buffer[idx] = value
        self._bin_sums[bin_idx] = np.sum(self._bin(bin_idx))
        if needs_recompute_min:
            self._recompute_min()

    def _bin(self, bin_idx):
        if bin_idx == len(self._bin_sums) - 1:
            return self._buffer[self._bin_size * bin_idx:]
        return self._buffer[self._bin_size * bin_idx: self._bin_size * (bin_idx + 1)]

    def _recompute_min(self):
        if self._used < self._capacity:
            self._min_id = np.argmin(self._buffer[:self._used])
        else:
            self._min_id = np.argmin(self._buffer)

        self._min = self._buffer[self._min_id]
