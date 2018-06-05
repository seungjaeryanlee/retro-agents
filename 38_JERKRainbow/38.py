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

from sonic_util import AllowBacktracking, make_envs

from jerk import run_jerk


def main():
    """Run DQN until the environment throws an exception."""
    jerk_env, rainbow_env = make_envs(stack=False, scale_rew=False)
    rainbow_env = AllowBacktracking(rainbow_env)
    rainbow_env = BatchedFrameStack(BatchedGymEnv([[rainbow_env]]), num_images=4, concat=False)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    with tf.Session(config=config) as sess:
        dqn = DQN(*rainbow_models(sess,
                                  rainbow_env.action_space.n,
                                  gym_space_vectorizer(rainbow_env.observation_space),
                                  min_val=-200,
                                  max_val=200))
        player = NStepPlayer(BatchedPlayer(rainbow_env, dqn.online_net), 3)
        optimize = dqn.optimize(learning_rate=1e-4)
        sess.run(tf.global_variables_initializer())

        # Populate Replay buffer with JERK
        replay_buffer = PrioritizedReplayBuffer(500000, 0.5, 0.4, epsilon=0.1)
        run_jerk(jerk_env, replay_buffer, 20000)

        dqn.train(num_steps=2000000, # Make sure an exception arrives before we stop.
                  player=player,
                  replay_buffer=replay_buffer,
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
