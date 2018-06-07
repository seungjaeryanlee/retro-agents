"""
From https://github.com/unixpickle/anyrl-py/blob/master/anyrl/rollouts/replay.py
"""
import numpy as np
from math import sqrt

class FloatBuffer:
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

    def set_value(self, idx, value):
        """Set the value at the given index."""
        idx = (idx + self._start) % self._capacity
        self._set_idx(idx, value)

    def min(self):
        """Get the minimum value in the buffer."""
        return self._min

    def sum(self):
        """Get the sum of the values in the buffer."""
        return np.sum(self._bin_sums)

    def _set_idx(self, idx, value):
        assert not np.isnan(value)
        assert value > 0
        needs_recompute = False
        if self._min == self._buffer[idx]:
            needs_recompute = True
        elif value < self._min:
            self._min = value
        bin_idx = idx // self._bin_size
        self._buffer[idx] = value
        self._bin_sums[bin_idx] = np.sum(self._bin(bin_idx))
        if needs_recompute:
            self._recompute_min()

    def _bin(self, bin_idx):
        if bin_idx == len(self._bin_sums) - 1:
            return self._buffer[self._bin_size * bin_idx:]
        return self._buffer[self._bin_size * bin_idx: self._bin_size * (bin_idx + 1)]

    def _recompute_min(self):
        if self._used < self._capacity:
            self._min = np.min(self._buffer[:self._used])
        else:
            self._min = np.min(self._buffer)
