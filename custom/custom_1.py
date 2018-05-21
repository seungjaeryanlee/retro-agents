#!/usr/bin/env python

"""
Train an agent on Sonic using an open source Rainbow DQN
implementation.
"""
import tensorflow as tf

from anyrl.algos import DQN
from anyrl.envs import BatchedGymEnv
from anyrl.envs.wrappers import BatchedFrameStack
from anyrl.models import rainbow_models
from anyrl.rollouts import BatchedPlayer, PrioritizedReplayBuffer, NStepPlayer
from anyrl.spaces import gym_space_vectorizer
import gym_remote.exceptions as gre

from sonic_util import AllowBacktracking, make_env

import numpy as np
from math import sqrt
# Comment out assertion that value > 0
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

    def reverse_sample(self, num_values):
        assert self._used >= num_values
        res = []
        probs = []
        reversed_bin_sums = np.divide(1, self._bin_sums)
        bin_probs = reversed_bin_sums / np.sum(np.divide(1, reversed_bin_sums))
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
        # assert value > 0
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

class CustomPrioritizedReplayBuffer(PrioritizedReplayBuffer):
    def __init__(self, capacity, alpha, beta, first_max=1, epsilon=0):
        super().__init__(capacity, alpha, beta, first_max, epsilon)
        self.errors = CustomFloatBuffer(capacity)

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
            indices, _ = self.errors.reverse_sample(1)
            self.transitions.pop(indices[0])

def main():
    """Run DQN until the environment throws an exception."""
    env = AllowBacktracking(make_env(stack=False, scale_rew=False))
    env = BatchedFrameStack(BatchedGymEnv([[env]]), num_images=4, concat=False)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    with tf.Session(config=config) as sess:
        dqn = DQN(*rainbow_models(sess,
                                  env.action_space.n,
                                  gym_space_vectorizer(env.observation_space),
                                  min_val=-200,
                                  max_val=200))
        player = NStepPlayer(BatchedPlayer(env, dqn.online_net), 3)
        optimize = dqn.optimize(learning_rate=1e-3)
        sess.run(tf.global_variables_initializer())
        dqn.train(num_steps=2000000, # Make sure an exception arrives before we stop.
                  player=player,
                  replay_buffer=CustomPrioritizedReplayBuffer(500000, 0.7, 0.4, epsilon=0),
                  optimize_op=optimize,
                  train_interval=1,
                  target_interval=8192,
                  batch_size=32,
                  min_buffer_size=20000)

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)
