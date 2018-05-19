# Versions of Agents

1. [simple-agent.py](https://contest.openai.com/details)
   * Score: 1345.55
   * Task Scores: 802.44, 1214.36, 2294.27, 1770.17, 646.49
2. [ppo2_agent.py](https://github.com/openai/retro-baselines/blob/master/agents/ppo2_agent.py)
   * Score: 2481.84
   * Task Scores: 2991.82, 2559.34, 2898.30, 2038.58, 1921.16
3. [rainbow_agent.py](https://github.com/openai/retro-baselines/blob/master/agents/rainbow_agent.py)
   * Score: 1358.18
   * Task Scores: 769.13, 1232.46, 2400.22, 1729.80, 659.30
4. [rainbow_agent.py](https://github.com/openai/retro-baselines/blob/master/agents/rainbow_agent.py)
   * Score: 1349.99
   * Task Scores: 747.69, 1196.37, 2380.21, 1771.47, 654.21
5. [5_rainbow.py](https://github.com/seungjaeryanlee/retro-agents/blob/master/5_rainbow.py)
   - Score: 3849.35
   - Task Scores: 7414.56, 2917.43, 3484.82, 2695.24, 2734.69
   - learning_rate=1e-3
6. [6_rainbow.py](https://github.com/seungjaeryanlee/retro-agents/blob/master/6_rainbow.py)
   - Score: 2447.30
   - Task Scores: 5149.68, 2255.74, 1437.19, 1979.20, 1414.68
   - learning_rate=1e-2
7. [7_rainbow.py](https://github.com/seungjaeryanlee/retro-agents/blob/master/7_rainbow.py)
   - Score: 4258.93*
   - Task Scores: 7523.93, 3021.56, 3003.30, 4683.19*, 3062.69
   - #4 was finished (4683.19) and restarted for unknown reason, and the second time it gave worse performance (1960.31).
   - learning_rate=1e-3, replay_buffer_alpha=0.7
8. [8_rainbow.py](https://github.com/seungjaeryanlee/retro-agents/blob/master/8_rainbow.py)
   - Score: 3535.15
   - Task Scores: 7597.51, 1588.43, 3422.14, 2433.19, 2634.49
   - learning_rate=1e-3, replay_buffer_alpha=0.8
9. [9_rainbow.py](https://github.com/seungjaeryanlee/retro-agents/blob/master/9_rainbow.py)
   - Score: 3396.40
   - Task Scores: 7205.39, 2920.50, 2406.80, 1832.62, 2616.69
   - learning_rate=1e-3, replay_buffer_alpha=0.7, num_images=5
10. [10_rainbow.py](https://github.com/seungjaeryanlee/retro-agents/blob/master/10_rainbow.py)
    - Score: 3470.62
    - Task Scores: 7448.09, 2597.23, 2659.62, 2192.36, 2455.79
    - learning_rate=1e-3, replay_buffer_alpha=0.7, num_images=3
11. [11_rainbow.py](https://github.com/seungjaeryanlee/retro-agents/blob/master/11_rainbow.py)
    - Score: 3529.88
    - Task Scores: 6959.38, 3106.55, 1693.94, 4267.29, 1622.26
    - learning_rate=1e-3, replay_buffer_alpha=0.7, sigma0: 0.1
12. [12_rainbow.py](https://github.com/seungjaeryanlee/retro-agents/blob/master/12_rainbow.py)
    - error due to assertion (value > 0) in FloatBuffer: value = (weights + epsilon) ** ?
    - learning_rate=1e-3, replay_buffer_alpha=0.7, sigma0: 0.1, epsilon = 0
14. [14_rainbow.py](https://github.com/seungjaeryanlee/retro-agents/blob/master/14_rainbow.py)
    - Score: 3657.89
    - Task Scores: 7317.67, 2453.19, 2003.89, 4310.35, 2204.33
    - learning_rate=1e-3, replay_buffer_alpha=0.7, sigma0: 0.1, epsilon = 1e-5
15. [15_rainbow.py](https://github.com/seungjaeryanlee/retro-agents/blob/master/15_rainbow.py)
    - Score: TBD
    - Task Scores: TBD
    - learning_rate=1e-3, replay_buffer_alpha=0.7, NStep: 5
16. [16_rainbow.py](https://github.com/seungjaeryanlee/retro-agents/blob/master/16_rainbow.py)
    - Score: TBD
    - Task Scores: TBD
    - learning_rate=1e-3, replay_buffer_alpha=0.7, NStep: 1
17. [17_rainbow.py](https://github.com/seungjaeryanlee/retro-agents/blob/master/17_rainbow.py)
    - Score: TBD
    - Task Scores: TBD
    - learning_rate=1e-3, replay_buffer_alpha=0.7, NStep: 3, epsilon=0

 - clr_rainbow_1.py
    - Use get_clr(), step_size=20000, N_step=5, alpha=0.7
    - ETA keeps increasing
 - clr_rainbow_2.py
    - Use get_clr2(), step_size=20000, N_step=5, alpha=0.7
    - ETA keeps increasing
 - clr_rainbow_3.py
    - get_clr2(), step_size=1000, N_step=3, alpha=0.7

# Idea Bank
 - Vary learning rate or epsilon by time
 - Try different [Adam learning rate / epsilon](https://github.com/unixpickle/anyrl-py/blob/531dd920e77f1b77d63d52bd56aad0807bfdccd8/anyrl/algos/dqn.py)
 - Read [Rainbow paper](https://arxiv.org/pdf/1710.02298.pdf)
   - Increase replay memory buffer size (20K -> 80K to follow paper)
   - Increase beta over time (to follow paper)
   - Change NStepPlayer step size (known to be sensitive)
   - With noisynet, set epsilon = 0
   - Without noisynet, Decrease epsilon over time (1 -> 0)

# Checked Ideas
**Bold** for best result, *Italics* for initial value.
 - Change Adam Learning Rate
   - 1e-2, **1e-3**, *1e-4*
 - Change Priority Exponent
   - *0.5*, **0.7**, 0.8
 - Change Number of Images
   - 3, **4**, 5
 - Change initial Noisy Net noise (sigma0)
   - 0.1, *0.5*
 - Decrease Exploration Epsilon in PrioritizedReplayBuffer to 0
   - 1e-5, *1e-1*
 - Change NStepPlayer step size
   - 1, *3*, 5
