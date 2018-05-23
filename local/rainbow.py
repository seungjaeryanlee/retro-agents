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
import time
import random

from sonic_util import AllowBacktracking, make_local_env
from SaveDQN import SaveDQN

def load_train_save(game, state):
    """Run DQN until the environment throws an exception."""
    env = make_local_env(game=game, state=state)
    env = AllowBacktracking(env)
    env = BatchedFrameStack(BatchedGymEnv([[env]]), num_images=4, concat=False)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101

    with tf.Session(config=config) as sess:
        dqn = SaveDQN(*rainbow_models(sess,
                                  env.action_space.n,
                                  gym_space_vectorizer(env.observation_space),
                                  min_val=-200,
                                  max_val=200))
        dqn.load()

        # print(dqn.base)
        player = NStepPlayer(BatchedPlayer(env, dqn.online_net), 3)
        optimize = dqn.optimize(learning_rate=1e-3)
        sess.run(tf.global_variables_initializer())
        dqn.train(num_steps=100000, # Make sure an exception arrives before we stop.
                  player=player,
                  replay_buffer=PrioritizedReplayBuffer(500000, 0.7, 0.4, epsilon=0.1),
                  optimize_op=optimize,
                  train_interval=1,
                  target_interval=8192,
                  batch_size=32,
                  min_buffer_size=20000)
        # 10K steps: 45s
        dqn.save()

def main():
    train_tuples = [
        ('SonicTheHedgehog-Genesis', 'LabyrinthZone.Act1'),
        ('SonicTheHedgehog-Genesis', 'LabyrinthZone.Act2'),
        ('SonicTheHedgehog-Genesis', 'LabyrinthZone.Act3'),
        ('SonicTheHedgehog-Genesis', 'SpringYardZone.Act2'),
        ('SonicTheHedgehog-Genesis', 'SpringYardZone.Act3'),
        ('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1'),
        ('SonicTheHedgehog-Genesis', 'GreenHillZone.Act3'),
        ('SonicTheHedgehog-Genesis', 'StarLightZone.Act1'),
        ('SonicTheHedgehog-Genesis', 'StarLightZone.Act2'),
        ('SonicTheHedgehog-Genesis', 'MarbleZone.Act1'),
        ('SonicTheHedgehog-Genesis', 'MarbleZone.Act2'),
        ('SonicTheHedgehog-Genesis', 'MarbleZone.Act3'),
        ('SonicTheHedgehog-Genesis', 'ScrapBrainZone.Act2'),
    ]
    game, state = random.choice(train_tuples)
    print('--------------------------------------------------')
    print('Training with Game {} Zone {}'.format(game, state))
    print('--------------------------------------------------')
    load_train_save(game=game, state=state)

if __name__ == '__main__':
    main()