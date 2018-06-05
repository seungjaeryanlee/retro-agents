import random
import numpy as np
import gym

EXPLOIT_BIAS = 0.25
TOTAL_TIMESTEPS = int(1e6)

def run_jerk(env, replay_buffer, n_steps):
    new_ep = True
    solutions = []
    steps = 0
    while True:
        if new_ep:
            if (solutions and random.random() < EXPLOIT_BIAS + env.total_steps_ever / TOTAL_TIMESTEPS):
                solutions = sorted(solutions, key=lambda x: np.mean(x[0]))
                best_pair = solutions[-1]
                new_rew = exploit(env, replay_buffer, best_pair[1])
                best_pair[0].append(new_rew)
                continue
            else:
                env.reset()
                new_ep = False
        rew, new_ep = move(env, replay_buffer, 100)
        steps += 100

        if new_ep and steps >= n_steps:
            break

        if not new_ep and rew <= 0:
            _, new_ep = move(env, replay_buffer, 70, left=True)
            steps += 70
        if new_ep:
            solutions.append(([max(env.reward_history)], env.best_sequence()))

        if new_ep and steps >= n_steps:
            break

def move(env, replay_buffer, num_steps, left=False, jump_prob=1.0 / 10.0, jump_repeat=4):
    """
    Move right or left for a certain number of steps,
    jumping periodically.
    """
    total_rew = 0.0
    done = False
    steps_taken = 0
    jumping_steps_left = 0
    while not done and steps_taken < num_steps:
        action = np.zeros((12,), dtype=np.bool)
        action[6] = left
        action[7] = not left
        if jumping_steps_left > 0:
            action[0] = True
            jumping_steps_left -= 1
        else:
            if random.random() < jump_prob:
                jumping_steps_left = jump_repeat - 1
                action[0] = True
        obs, rew, done, _ = env.step(action)
        total_rew += rew
        
        # TODO: Add left B, right B to Discretizer?
        if action[0] and action[6]:
            discrete_action = 7
        elif action[0] and action[7]:
            discrete_action = 7
        elif action[6]: # LEFT
            discrete_action = 0
        elif action[7]: # RIGHT
            discrete_action = 1
        else: # NOOP - should never happen
            discrete_action = 4 # DOWN

        # CUSTOM: Check feed_dict in https://github.com/unixpickle/anyrl-py/blob/master/anyrl/algos/dqn.py
        replay_buffer.add_sample({
            'obs': obs,
            'model_outs': {'actions': [discrete_action]},
            'rewards': [rew],
            'new_obs': (obs if not done else None),
            # 'info': info,
            # 'start_state': _reduce_states(self._cur_states[sub_batch], i),
            # 'episode_id': self._episode_ids[sub_batch][i],
            # 'episode_step': self._episode_steps[sub_batch][i],
            # 'end_time': end_time,
            'is_last': done,
            'total_reward': total_rew,
        })

        steps_taken += 1
        if done:
            break
    return total_rew, done

def exploit(env, replay_buffer, sequence):
    """
    Replay an action sequence; pad with NOPs if needed.

    Returns the final cumulative reward.
    """
    env.reset()
    done = False
    idx = 0
    while not done:
        if idx >= len(sequence):
            action = np.zeros((12,), dtype='bool')
        else:
            action = sequence[idx]

        obs, rew, done, _ = env.step(action)
        idx += 1

        # TODO: Add left B, right B to Discretizer?
        if action[0] and action[6]:
            discrete_action = 7
        elif action[0] and action[7]:
            discrete_action = 7
        elif action[6]: # LEFT
            discrete_action = 0
        elif action[7]: # RIGHT
            discrete_action = 1
        else: # NOOP
            discrete_action = 4 # DOWN

        replay_buffer.add_sample({
            'obs': obs,
            'model_outs': {'actions': [discrete_action]},
            'rewards': [rew],
            'new_obs': (obs if not done else None),
            # 'info': info,
            # 'start_state': _reduce_states(self._cur_states[sub_batch], i),
            # 'episode_id': self._episode_ids[sub_batch][i],
            # 'episode_step': self._episode_steps[sub_batch][i],
            # 'end_time': end_time,
            'is_last': done,
            'total_reward': env.total_reward,
        })
    return env.total_reward

class TrackedEnv(gym.Wrapper):
    """
    An environment that tracks the current trajectory and
    the total number of timesteps ever taken.
    """
    def __init__(self, env):
        super(TrackedEnv, self).__init__(env)
        self.action_history = []
        self.reward_history = []
        self.total_reward = 0
        self.total_steps_ever = 0

    def best_sequence(self):
        """
        Get the prefix of the trajectory with the best
        cumulative reward.
        """
        max_cumulative = max(self.reward_history)
        for i, rew in enumerate(self.reward_history):
            if rew == max_cumulative:
                return self.action_history[:i+1]
        raise RuntimeError('unreachable')

    # pylint: disable=E0202
    def reset(self, **kwargs):
        self.action_history = []
        self.reward_history = []
        self.total_reward = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        self.total_steps_ever += 1
        self.action_history.append(action.copy())
        obs, rew, done, info = self.env.step(action)
        self.total_reward += rew
        self.reward_history.append(self.total_reward)
        return obs, rew, done, info