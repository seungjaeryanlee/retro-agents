# Custom

## Scores

 * Baseline
    * Score: 1358.18 (769.13, 1232.46, 2400.22, 1729.80, 659.30)
 * Tuned Baseline (v16)
    * Score: 3919.81 (7567.57, 3209.94, 3528.58, 2110.28, 3182.64)
 * "Tuned" Uniform Experience Replay (v3)
    * Score: 3848.98 (7566.88, 2977.46, 3324.90, 1993.85, 3391.80)
 * "Tuned" Moving Average Error Threshold (v1)
    * Score: 3710.31 (7595.58, 2715.25, 2872.21, 2343.48, 3025.02)
 * "Tuned" Moving Average Error Threshold with Exponential Decay (v2)
    * Score: 3769.19 (7466.05, 3203.78, 2952.01, 2209.81, 3014.29)
 * "Tuned" Minimum Error Threshold (v4)
    * Score: 3808.20 (7711.53, 3195.36, 3362.40, 2098.70, 2672.99)
 * "Tuned" Minimum Error Threshold with smaller cache (v5)
    * Score: TBD
    * capacity: 500000 -> 125000
    * min_buffer: 20000 -> 10000
 * "Tuned" Minimum Error Threshold when Full (v6)
    * Score: TBD



 * "Tuned" Moving Average Error Threshold with Exponential Decay and Reset
    * Score: TBD
    * Learning rate: 1e-4 -> 1e-3
    * Alpha: 0.5 -> 0.7
 * "Tuned" Weighted Moving Average Error Threshold with Exponential Decay and Reset
    * Score: TBD
    * Learning rate: 1e-4 -> 1e-3
    * Alpha: 0.5 -> 0.7
 * "Tuned" Exponential Moving Average Error Threshold with Exponential Decay and Reset
    * Score: TBD
    * Learning rate: 1e-4 -> 1e-3
    * Alpha: 0.5 -> 0.7

## Ideas
 * Re-tune Hyperparameters for custom agents
 * Decrease Replay Memory & Minimum Threshold
 * Train & Save weights from local computer
