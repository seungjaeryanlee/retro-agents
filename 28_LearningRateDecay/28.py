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


def get_optimize_operation(sess, dqn, learning_rate=6.25e-5, epsilon=1.5e-4, **adam_kwargs):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon, **adam_kwargs)
    sess.run(tf.variables_initializer(optimizer.variables()))
    return optimizer.minimize(dqn.loss)

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
        sess.run(tf.global_variables_initializer())

        replay = PrioritizedReplayBuffer(500000, 0.5, 0.4, epsilon=0.1)

        # Crude implementation of Learning Rate Decay
        for i in range(1, 10):
            optimize = get_optimize_operation(sess, dqn, learning_rate=1e-3 / i)
            dqn.train(num_steps=100000,
                      player=player,
                      replay_buffer=replay,
                      optimize_op=optimize,
                      train_interval=1,
                      target_interval=8192,
                      batch_size=32,
                      min_buffer_size=20000)

        optimize = get_optimize_operation(sess, dqn, learning_rate=1e-4)
        dqn.train(num_steps=1000000,
                player=player,
                replay_buffer=replay,
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