#!/usr/bin/env python

"""
Train an agent on Sonic using an open source Rainbow DQN
implementation.
"""

import time
import tensorflow as tf
from collections import OrderedDict

# from anyrl.algos import DQN
from anyrl.envs import BatchedGymEnv
from anyrl.envs.wrappers import BatchedFrameStack
from anyrl.models import rainbow_models
# from anyrl.rollouts import BatchedPlayer, PrioritizedReplayBuffer, NStepPlayer
from anyrl.rollouts import PrioritizedReplayBuffer, Player, BatchedPlayer
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

    # Populate Replay buffer with JERK
    replay_buffer = PrioritizedReplayBuffer(500000, 0.5, 0.4, epsilon=0.1)
    run_jerk(jerk_env, replay_buffer, 20000)
    print('Replay Buffer Size: ', replay_buffer.size())

    with tf.Session(config=config) as sess:
        dqn = DQN(*rainbow_models(sess,
                                  rainbow_env.action_space.n,
                                  gym_space_vectorizer(rainbow_env.observation_space),
                                  min_val=-200,
                                  max_val=200))
        player = NStepPlayer(BatchedPlayer(rainbow_env, dqn.online_net), 3)
        optimize = dqn.optimize(learning_rate=1e-4)
        sess.run(tf.global_variables_initializer())
        dqn.train(num_steps=2000000, # Make sure an exception arrives before we stop.
                  player=player,
                  replay_buffer=replay_buffer,
                  optimize_op=optimize,
                  train_interval=1,
                  target_interval=8192,
                  batch_size=32,
                  min_buffer_size=20000)


class DQN:
    """
    Train TFQNetwork models using Q-learning.
    """

    def __init__(self, online_net, target_net, discount=0.99):
        """
        Create a Q-learning session.
        Args:
          online_net: the online TFQNetwork.
          target_net: the target TFQNetwork.
          discount: the per-step discount factor.
        """
        self.online_net = online_net
        self.target_net = target_net
        self.discount = discount

        obs_shape = (None,) + online_net.obs_vectorizer.out_shape
        self.obses_ph = tf.placeholder(online_net.input_dtype, shape=obs_shape)
        self.actions_ph = tf.placeholder(tf.int32, shape=(None,))
        self.rews_ph = tf.placeholder(tf.float32, shape=(None,))
        self.new_obses_ph = tf.placeholder(online_net.input_dtype, shape=obs_shape)
        self.terminals_ph = tf.placeholder(tf.bool, shape=(None,))
        self.discounts_ph = tf.placeholder(tf.float32, shape=(None,))
        self.weights_ph = tf.placeholder(tf.float32, shape=(None,))

        losses = online_net.transition_loss(target_net, self.obses_ph, self.actions_ph,
                                            self.rews_ph, self.new_obses_ph, self.terminals_ph,
                                            self.discounts_ph)
        self.losses = self.weights_ph * losses
        self.loss = tf.reduce_mean(self.losses)

        assigns = []
        for dst, src in zip(target_net.variables, online_net.variables):
            assigns.append(tf.assign(dst, src))
        self.update_target = tf.group(*assigns)

    def feed_dict(self, transitions):
        """
        Generate a feed_dict that feeds the batch of
        transitions to the DQN loss terms.
        Args:
          transition: a sequence of transition dicts, as
            defined in anyrl.rollouts.ReplayBuffer.
        Returns:
          A dict which can be fed to tf.Session.run().
        """
        obs_vect = self.online_net.obs_vectorizer
        res = {
            self.obses_ph: obs_vect.to_vecs([t['obs'] for t in transitions]),
            self.actions_ph: [t['model_outs']['actions'][0] for t in transitions],
            self.rews_ph: [self._discounted_rewards(t['rewards']) for t in transitions],
            self.terminals_ph: [t['new_obs'] is None for t in transitions],
            self.discounts_ph: [(self.discount ** len(t['rewards'])) for t in transitions],
            self.weights_ph: [t['weight'] for t in transitions]
        }
        new_obses = []
        for trans in transitions:
            if trans['new_obs'] is None:
                new_obses.append(trans['obs'])
            else:
                new_obses.append(trans['new_obs'])
        res[self.new_obses_ph] = obs_vect.to_vecs(new_obses)
        return res

    def optimize(self, learning_rate=6.25e-5, epsilon=1.5e-4, **adam_kwargs):
        """
        Create a TF Op that optimizes the objective.
        Args:
          learning_rate: the Adam learning rate.
          epsilon: the Adam epsilon.
        """
        optim = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon, **adam_kwargs)
        return optim.minimize(self.loss)

    def train(self,
              num_steps,
              player,
              replay_buffer,
              optimize_op,
              train_interval=1,
              target_interval=8192,
              batch_size=32,
              min_buffer_size=20000,
              tf_schedules=(),
              handle_ep=lambda steps, rew: None,
              timeout=None):
        """
        Run an automated training loop.
        This is meant to provide a convenient way to run a
        standard training loop without any modifications.
        You may get more flexibility by writing your own
        training loop.
        Args:
          num_steps: the number of timesteps to run.
          player: the Player for gathering experience.
          replay_buffer: the ReplayBuffer for experience.
          optimize_op: a TF Op to optimize the model.
          train_interval: timesteps per training step.
          target_interval: number of timesteps between
            target network updates.
          batch_size: the size of experience mini-batches.
          min_buffer_size: minimum replay buffer size
            before training is performed.
          tf_schedules: a sequence of TFSchedules that are
            updated with the number of steps taken.
          handle_ep: called with information about every
            completed episode.
          timeout: if set, this is a number of seconds
            after which the training loop should exit.
        """
        print('Entered dqn.train()')
        sess = self.online_net.session
        sess.run(self.update_target)
        steps_taken = 0
        next_target_update = target_interval
        next_train_step = train_interval
        start_time = time.time()
        while steps_taken < num_steps:
            print('Inside while loop')
            if timeout is not None and time.time() - start_time > timeout:
                print('Timeout Conditional')
                return
            print('Player: ', player)
            transitions = player.play()
            print('Player played')
            print('Transitions: ', transitions)
            for trans in transitions:
                print('Inside transition loop')
                if trans['is_last']:
                    handle_ep(trans['episode_step'] + 1, trans['total_reward'])
                replay_buffer.add_sample(trans)
                steps_taken += 1
                for sched in tf_schedules:
                    sched.add_time(sess, 1)
                if replay_buffer.size >= min_buffer_size and steps_taken >= next_train_step:
                    print('Replay Buffer over min buffer size')
                    next_train_step = steps_taken + train_interval
                    batch = replay_buffer.sample(batch_size)
                    _, losses = sess.run((optimize_op, self.losses),
                                         feed_dict=self.feed_dict(batch))
                    replay_buffer.update_weights(batch, losses)
                if steps_taken >= next_target_update:
                    print('Time to update target')
                    next_target_update = steps_taken + target_interval
                    sess.run(self.update_target)
        print('Exited dqn.train()')

    def _discounted_rewards(self, rews):
        res = 0
        for i, rew in enumerate(rews):
            res += rew * (self.discount ** i)
        return res

class NStepPlayer(Player):
    """
    A Player that wraps another Player and uses n-step
    transitions instead of 1-step transitions.
    """

    def __init__(self, player, num_steps):
        self.player = player
        self.num_steps = num_steps
        self._ep_to_history = OrderedDict()

    def play(self):
        # Let the buffers fill up until we get actual
        # n-step transitions.
        print('NStepPlayer.play() called')
        while True:
            transes = self._play_once()
            if transes:
                print('transes is True')
                return transes

    def _play_once(self):
        print('NStepPlayer._play_once() called')
        for trans in self.player.play():
            print('self.player.play() called: in for loop for trans')
            assert len(trans['rewards']) == 1
            ep_id = trans['episode_id']
            if ep_id in self._ep_to_history:
                self._ep_to_history[ep_id].append(trans)
            else:
                self._ep_to_history[ep_id] = [trans]
        print('Finished for loop')
        res = []
        for ep_id, history in list(self._ep_to_history.items()):
            print('In second for loop')
            while history:
                print('While history loop')
                trans = self._next_transition(history)
                if trans is None:
                    print('trans is None')
                    break
                res.append(trans)
            if not history:
                del self._ep_to_history[ep_id]
        print('Finished _play_once() - returning res: ', res)
        return res

    def _next_transition(self, history):
        if len(history) < self.num_steps:
            if not history[-1]['is_last']:
                return None
        res = history[0].copy()
        res['rewards'] = [h['rewards'][0] for h in history[:self.num_steps]]
        res['total_reward'] += sum(h['rewards'][0] for h in history[1:self.num_steps])
        if len(history) >= self.num_steps:
            res['new_obs'] = history[self.num_steps-1]['new_obs']
        else:
            res['new_obs'] = None
        del history[0]
        return res

if __name__ == '__main__':
    try:
        main()
    # except gre.GymRemoteError as exc:
    except Exception as exc:
        print('exception', exc)
