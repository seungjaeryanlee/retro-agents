"""
The core implementation of deep Q-learning.
"""

import time
import tensorflow as tf

from anyrl.algos import DQN


class SaveDQN(DQN):
    """
    Train TFQNetwork models using Q-learning.
    """
    def load(self):
        saver = tf.train.Saver()
        saver.restore(self.online_net.session, "model/rainbow.ckpt")

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.online_net.session, "model/rainbow.ckpt")
