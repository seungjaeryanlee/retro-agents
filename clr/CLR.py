
import time
import numpy as np
import tensorflow as tf
from anyrl.algos import DQN

class CLRDQN(DQN):
    def get_clr(self, iterations, step_size=1000, base_lr = 1e-5, max_lr = 1e-4):
        """
        Cyclical learning Rate (Triangular)
        From https://github.com/bckenstler/CLR
        """
        cycle = np.floor(1 + iterations / (2 * step_size))
        x = np.abs(iterations / step_size - 2*cycle + 1)
        return base_lr + (max_lr - base_lr) * np.maximum(0, (1-x))

    def get_clr2(self, iterations, step_size=1000, base_lr = 1e-5, max_lr = 1e-4):
        """
        Cyclical learning Rate (Triangular 2)
        From https://github.com/bckenstler/CLR
        """
        cycle = np.floor(1 + iterations / (2 * step_size))
        x = np.abs(iterations / step_size - 2*cycle + 1)
        return base_lr + (max_lr - base_lr) * np.maximum(0, (1-x))/float(2**(cycle-1))


    def clr_optimize(self, learning_rate=6.25e-5, epsilon=1.5e-4, **adam_kwargs):
        """
        Create a TF Op that optimizes the objective.
        Args:
          learning_rate: the Adam learning rate.
          epsilon: the Adam epsilon.
        """
        self.learning_rate_var = tf.Variable(learning_rate, name='learning_rate')
        optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate_var, epsilon=epsilon, **adam_kwargs)
        self.optimize_op = optim.minimize(self.loss)
    
    def clr_train(self,
              num_steps,
              player,
              replay_buffer,
              clr_type=1,
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
        sess = self.online_net.session
        sess.run(self.update_target)
        steps_taken = 0
        next_target_update = target_interval
        next_train_step = train_interval
        start_time = time.time()
        while steps_taken < num_steps:
            if timeout is not None and time.time() - start_time > timeout:
                return
            transitions = player.play()
            for trans in transitions:
                if trans['is_last']:
                    handle_ep(trans['episode_step'] + 1, trans['total_reward'])
                replay_buffer.add_sample(trans)
                steps_taken += 1
                for sched in tf_schedules:
                    sched.add_time(sess, 1)
                if replay_buffer.size >= min_buffer_size and steps_taken >= next_train_step:
                    if clr_type == 1:
                        assign_op = self.learning_rate_var.assign(self.get_clr(steps_taken))
                    elif clr_type == 2:
                        assign_op = self.learning_rate_var.assign(self.get_clr2(steps_taken))
                    sess.run(assign_op)
                    next_train_step = steps_taken + train_interval
                    batch = replay_buffer.sample(batch_size)
                    _, losses = sess.run((self.optimize_op, self.losses),
                                         feed_dict=self.feed_dict(batch))
                    replay_buffer.update_weights(batch, losses)
                if steps_taken >= next_target_update:
                    next_target_update = steps_taken + target_interval
                    sess.run(self.update_target)
