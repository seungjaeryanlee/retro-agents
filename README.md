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
   - epsilon=1e-3
6. [6_rainbow.py](https://github.com/seungjaeryanlee/retro-agents/blob/master/6_rainbow.py)
   - Score: 2447.30
   - Task Scores: 5149.68, 2255.74, 1437.19, 1979.20, 1414.68
   - epsilon=1e-2
7. [7_rainbow.py](https://github.com/seungjaeryanlee/retro-agents/blob/master/7_rainbow.py)
   - Score: 4258.93*
   - Task Scores: 7523.93, 3021.56, 3003.30, 4683.19*, 3062.69
   - #4 was finished (4683.19) and restarted for unknown reason, and the second time it gave worse performance (1960.31).
   - epsilon=1e-3, replay_buffer_alpha=0.7
8. [8_rainbow.py](https://github.com/seungjaeryanlee/retro-agents/blob/master/8_rainbow.py)
   - Score: 3535.15
   - Task Scores: 7597.51, 1588.43, 3422.14, 2433.19, 2634.49
   - epsilon=1e-3, replay_buffer_alpha=0.8
9. [9_rainbow.py](https://github.com/seungjaeryanlee/retro-agents/blob/master/9_rainbow.py)
   - Score: TBD
   - Task Scores: TBD
   - epsilon=1e-3, replay_buffer_alpha=0.7, num_images=5
