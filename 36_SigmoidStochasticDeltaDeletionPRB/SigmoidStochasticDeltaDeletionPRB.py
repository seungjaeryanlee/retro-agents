import numpy as np
import random
from math import sqrt

from anyrl.rollouts import PrioritizedReplayBuffer
from FloatBuffer import FloatBuffer

class SigmoidStochasticDeltaDeletionPRB(PrioritizedReplayBuffer):
    """
    A prioritized replay buffer with minimum error deletion and
    loss-proportional sampling.
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
        When new samples are added without an explicit
        initial weight, the maximum weight argument ever
        seen is used. When the buffer is empty, first_max
        is used.
        """
        self.transitions.append(sample)
        if init_weight is None:
            self.errors.append(self._process_weight(self._max_weight_arg))
        else:
            self.errors.append(self._process_weight(init_weight))
        while len(self.transitions) > self.capacity:
            print('Deleting 1 because full')
            self.move_to_front(self.errors.min_delta_id_sample())
            del self.transitions[0]

    def move_to_front(self, error_idx):
        """
        Move transition, error and delta to front by switching places with first
        element. Done before deletion.
        """
        transition_idx = (error_idx - self.errors._start) % self.errors._capacity

        if transition_idx == 0:
            return

        # Switch transition
        temp = self.transitions[0]
        self.transitions[0] = self.transitions[transition_idx]
        self.transitions[transition_idx] = temp

        # Switch errors
        temp = self.errors._buffer[self.errors._start]
        self.errors._buffer[self.errors._start] = self.errors._buffer[error_idx]
        self.errors._buffer[error_idx] = temp

        # Switch deltas
        temp = self.errors._delta_buffer[self.errors._start]
        self.errors._delta_buffer[self.errors._start] = self.errors._delta_buffer[error_idx]
        self.errors._delta_buffer[error_idx] = temp

    def update_weights(self, samples, new_weights):
        samples_to_delete = []
        for sample, weight in zip(samples, new_weights):
            delete_sample = self.errors.set_value(sample['id'], self._process_weight(weight))
            if delete_sample:
                samples_to_delete.append(sample)

        # Delete chosen samples
        print('Deleting ', len(samples_to_delete))
        for sample in samples_to_delete:
            error_idx = (sample['id'] + self.errors._start) % self.errors._capacity
            self.move_to_front(error_idx)
            del self.transitions[0]
            self.errors._used -= 1

class CustomFloatBuffer(FloatBuffer):
    """A ring-buffer of floating point values."""

    def __init__(self, capacity, dtype='float64'):
        super().__init__(capacity, dtype)

        # Store deltas
        self._delta_buffer = np.full((capacity,), float('inf'), dtype=dtype)

    def set_value(self, idx, value):
        """Set the value at the given index."""
        idx = (idx + self._start) % self._capacity

        return self._set_idx_old(idx, value)

    def _set_idx_old(self, idx, value):
        """
        Set buffer[idx] = value for updated transition. This is NOT called
        in append() but is called in set_value(), which is only called in
        _update_weights() in PRB.
        """
        self._delta_buffer[idx] = value - self._buffer[idx]
        sigmoid_prob = 1 / (1 + np.exp(-1 * self._delta_buffer[idx]))
        print('Delta: ', self._delta_buffer[idx])
        print('Sigmoid: ', sigmoid_prob)

        super()._set_idx(idx, value)

        return np.random.random() < sigmoid_prob # Probability of deleting

    def min_delta_id_sample(self):
        """
        Sample index in reverse proportion to their delta.
        """
        e_neg_deltas = np.exp(-1 * self._delta_buffer)
        probs = e_neg_deltas / np.sum(e_neg_deltas)
        idx = np.random.choice(len(probs), p=probs)

        return idx
