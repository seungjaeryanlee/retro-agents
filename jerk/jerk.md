# JERK (Just Enough Retained Knowledge)

## Versions

 * Version 1
   * Score: 3968.25 (7684.04, 2437.97, 4233.26, 2497.53, 2988.46)
   * Baseline
 * Version 2
   * Score: 3964.28 (6528.07, 2520.57, 5120.44, 2550.38, 3101.96)
   * EXPLOIT_BIAS=0.5
 * Version 3
   * Score: 3532.72 (6386.37, 2340.73, 4248.20, 2345.35, 2342.96)
   * EXPLOIT_BIAS=0.125
 * Version 4
   * Score: 3570.10 (6401.60, 2321.92, 4194.45, 2399.12, 2533.41)
   * Squared Exploit Growth
 * Version 5
   * Score: TBD
   * NUM_STEPS = 400, BACKTRACK_NUM_STEPS = 140

## Hyperparameter Tuning

 * EXPLOIT_BIAS
   * 0.125
   * **0.25**
   * 0.5
 * EXPLOIT GROWTH
   * (No Growth)
   * **Linear Growth**
   * Squared Exploit Growth

## Ideas
 * Increase NUM_STEPS over time