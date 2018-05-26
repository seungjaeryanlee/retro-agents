import numpy as np
import random
from math import sqrt

from anyrl.rollouts import PrioritizedReplayBuffer

class DoubleSampling(PrioritizedReplayBuffer):
    """
    A replay buffer where some samples are sampled with priority and the rest
    are sampled uniformly.
    """

    def __init__(self, capacity, alpha, beta, first_max=1, epsilon=0, prb_rate=0.5):
        super().__init__(capacity, alpha, beta, first_max, epsilon)

        self.prb_rate = prb_rate

    def sample(self, num_samples):
        prb_num_samples = int(self.prb_rate * num_samples)
        urb_num_samples = num_samples - prb_num_samples

        return self.sample_prb(prb_num_samples) + self.sample_urb(urb_num_samples)

    def sample_prb(self, num_samples):
        indices, probs = self.errors.sample(num_samples)
        beta = float(self.beta)
        importance_weights = np.power(probs * self.size, -beta)
        importance_weights /= np.power(self.errors.min() / self.errors.sum() * self.size, -beta)
        samples = []
        for i, weight in zip(indices, importance_weights):
            sample = self.transitions[i].copy()
            sample['weight'] = weight
            sample['id'] = i
            samples.append(sample)
        return samples

    def sample_urb(self, num_samples):
        res = [random.choice(self.transitions).copy() for _ in range(num_samples)]
        for transition in res:
            transition['weight'] = 1
        return res
