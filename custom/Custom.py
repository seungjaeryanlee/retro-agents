import numpy as np

from anyrl.rollouts import PrioritizedReplayBuffer


class CachePrioritizedReplayBuffer(PrioritizedReplayBuffer):
    """
    A prioritized replay buffer with loss-proportional
    sampling.
    Weights passed to add_sample() and update_weights()
    are assumed to be error terms (e.g. the absolute TD
    error).
    """

    def __init__(self, capacity, alpha, beta, first_max=1, epsilon=0):
        """
        Create a prioritized replay buffer.
        The beta parameter can be any object that has
        support for the float() built-in.
        This way, you can use a TFScheduleValue.
        Args:
          capacity: the maximum number of transitions to
            store in the buffer.
          alpha: an exponent controlling the temperature.
            Higher values result in more prioritization.
            A value of 0 yields uniform prioritization.
          beta: an exponent controlling the amount of
            importance sampling. A value of 1 yields
            unbiased sampling. A value of 0 yields no
            importance sampling.
          first_max: the initial weight for new samples
            when no init_weight is specified and the
            buffer is completely empty.
          epsilon: a value which is added to every error
            term before the error term is used.
        """
        super().__init__(capacity, alpha, beta, first_max, epsilon)

        self.error_threshold = 0 # Threshold error for newly added sample

    def add_sample(self, sample, init_weight=None):
        """
        Add a sample to the buffer.
        When new samples are added without an explicit
        initial weight, the maximum weight argument ever
        seen is used. When the buffer is empty, first_max
        is used.
        """
        if init_weight is None:
            self.transitions.append(sample)
            self.errors.append(self._process_weight(self._max_weight_arg))
        else:
            new_error = self._process_weight(init_weight)
            if new_error < self.error_threshold:
                return

            self.transitions.append(sample)
            self.errors.append(new_error)

            # Update error_threshold via incremental average
            self.error_threshold += (new_error - self.error_threshold) / len(self.transitions)

        while len(self.transitions) > self.capacity:
            del self.transitions[0]

class CachePrioritizedReplayBuffer2(PrioritizedReplayBuffer):
    """
    A prioritized replay buffer with loss-proportional
    sampling.
    Weights passed to add_sample() and update_weights()
    are assumed to be error terms (e.g. the absolute TD
    error).
    """
    def __init__(self, capacity, alpha, beta, first_max=1, epsilon=0):
        """
        Create a prioritized replay buffer.
        The beta parameter can be any object that has
        support for the float() built-in.
        This way, you can use a TFScheduleValue.
        Args:
          capacity: the maximum number of transitions to
            store in the buffer.
          alpha: an exponent controlling the temperature.
            Higher values result in more prioritization.
            A value of 0 yields uniform prioritization.
          beta: an exponent controlling the amount of
            importance sampling. A value of 1 yields
            unbiased sampling. A value of 0 yields no
            importance sampling.
          first_max: the initial weight for new samples
            when no init_weight is specified and the
            buffer is completely empty.
          epsilon: a value which is added to every error
            term before the error term is used.
        """
        super().__init__(capacity, alpha, beta, first_max, epsilon)

        self.error_threshold = 0              # Threshold error for newly added sample
        self.error_threshold_decay_rate = 0.9 # Threshold decay after rejecting sample

    def add_sample(self, sample, init_weight=None):
        """
        Add a sample to the buffer.
        When new samples are added without an explicit
        initial weight, the maximum weight argument ever
        seen is used. When the buffer is empty, first_max
        is used.
        """
        if init_weight is None:
            self.transitions.append(sample)
            self.errors.append(self._process_weight(self._max_weight_arg))
        else:
            new_error = self._process_weight(init_weight)
            if new_error < self.error_threshold:
                self.error_threshold *= self.error_threshold_decay_rate
                return

            self.transitions.append(sample)
            self.errors.append(new_error)

            # Update error_threshold via incremental average
            self.error_threshold += (new_error - self.error_threshold) / len(self.transitions)

        while len(self.transitions) > self.capacity:
            del self.transitions[0]
