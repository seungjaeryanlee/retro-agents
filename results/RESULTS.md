# Results

This file contains the scores for all implemented agents. If the agent was tested mutliple times, it has multiple rows.

| Agent                                   | #1      | #2      | #3      | #4      | #5      | Average |
|-----------------------------------------|---------|---------|---------|---------|---------|---------|
| Baseline                                | 7662.17 | 2827.67 | 2740.57 | 5782.61 | 3339.20 | 4470.44 |
| AllAveragePRB                           | 7742.97 | 3481.13 | 2954.60 | 1366.88 | 3232.59 | 3755.63 |
| BufferAveragePRB                        | 7666.15 | 3745.61 | 1381.75 | 5005.94 | 3385.39 | 4236.97 |
| MinimumPRB                              | 7706.23 | 2569.26 | 3069.25 | 2513.12 | 3307.15 | 3833.00 |
| StochasticMaximumPRB                    | 7382.44 | 3739.57 | 2991.12 | 4902.19 | 3347.11 | 4472.48 |
| StochasticMaximumPRB                    | 7447.32 | 4030.97 | 2681.76 | 1907.79 | 2703.91 | 3754.35 |
| StochasticMaximumPRB                    | 7537.61 | 2970.10 | 2921.54 | 3514.32 | 2994.49 | 3987.61 |
| CombinedAveragePRB                      | 7593.99 | 2497.77 | 2981.47 | 5774.04 | 2705.14 | 4310.48 |
| Dual Sampling                           | 8070.37 | 2683.20 | 2494.67 | 3991.64 | 3104.40 | 4068.85 |
| BufferExponentialAveragePRB             | 7098.17 | 2828.04 | 1926.53 | 5857.17 | 3020.59 | 4146.10 |
| StochasticMaximumDeltaPRB               | 7655.47 | 2413.89 | 3521.59 | 5578.21 | 2869.85 | 4407.80 |
| MinimumDeletionPRB                      | 7717.11 | 2583.42 | 1387.13 | 5827.83 | 3414.06 | 4185.91 |
| NoDueling                               | 7796.19 | 3269.91 | 1435.77 | 2135.99 | 3441.13 | 3615.80 |
| UniformReplayBuffer                     | 7677.11 | 2446.46 | 3347.89 | 6351.07 | 1594.23 | 4282.15 |
| ShortTermMemoryPRB                      | 7711.45 | 2929.20 | 2167.58 | 1895.86 | 3119.19 | 3564.65 |
| StochasticMaximumURB                    | 7529.44 | 3086.36 | 2793.14 | 1006.20 | 3383.79 | 3559.79 |
| FullBufferStochasticMaximumPRB          | 7648.46 | 2380.47 | 2782.34 | 1823.71 | 3173.38 | 3561.67 |
| StochasticBufferAveragePRB              | 7563.53 | 2660.62 | 2359.74 | 4126.60 | 3231.24 | 3988.35 |
| StochasticDeletionPRB                   | 7426.93 | 2711.12 | 3251.27 | 4646.53 | 3447.60 | 4296.69 |
| NoMinBuffer                             | 7693.96 | 3126.24 | 2260.07 | 4282.86 | 3183.73 | 4108.97 |
| StochasticMaxStochasticDeletionPRB      | 7460.97 | 3860.93 | 2175.60 | 2426.37 | 3361.43 | 3857.06 |
| StochasticDeltaDeletionPRB              | 7797.27	| 3423.03 | 1003.70 | 1992.22 | 3486.24	| 3540.49 |
| StochasticMaxStochasticDeltaDeletionPRB | 7726.53 | 2638.33 | 2062.36 | 2975.46 | 3454.15 | 3771.37 |

Comparing the final score of StochasticMaximumPRB for Task #4 shows that the score has high variance. Based on the randomness, the same agent can have a score difference of about 3000 in the same environment.
