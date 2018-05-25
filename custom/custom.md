# Custom

## Scores

 * Baseline
    * Score: 1358.18 (769.13, 1232.46, 2400.22, 1729.80, 659.30)
 * Tuned Baseline (v16)
    * Score: 3919.81 (7567.57, 3209.94, 3528.58, 2110.28, 3182.64)
 * "Tuned" Uniform Experience Replay (v3)
    * Score: 3848.98 (7566.88, 2977.46, 3324.90, 1993.85, *3391.80)
 * "Tuned" Buffer Average PRB (v1)
    * Score: 3710.31 (7595.58, 2715.25, 2872.21, 2343.48, 3025.02)
 * "Tuned" Decaying Buffer Average PRB (v2)
    * Score: 3769.19 (7466.05, 3203.78, 2952.01, 2209.81, 3014.29)
 * "Tuned" Minimum Error PRB (v4)
    * Score: 3808.20 (7711.53, 3195.36, 3362.40, 2098.70, 2672.99)
 * "Tuned" Minimum Error PRB with smaller cache (v5)
    * Score: 3359.75 (7472.50, 2290.54, 1788.44, 2342.33, 2904.95)
    * capacity: 500000 -> 125000
    * min_buffer: 20000 -> 10000
 * "Tuned" All Average PRB (v6)
    * Score: 3969.89 (7236.84, 3350.97, 3237.88, 2732.87, 3290.87)
 * "Tuned" Stochastic Maximum PRB (v7)
    * Score: 3467.63 (7436.90, 3182.48, 1741.78, 1878.54, 3098.45)
 * "Tuned" All Average URB (v8)
    * Score: 3077.80 (7415.16, 2988.87, 410.87, 1832.16, 2741.93)
 * "Tuned" Decaying Buffer Average URB (v9)
    * Score: 3753.71 (7531.26, 3053.05, 2808.82, 1981.53, 3393.87)
 * "Tuned" All Exponential Average PRB (v10)
    * Score: 3772.42 (7564.60, 3105.25, 2963.96, 2133.30, 3095.01)
 * "Tuned" Undiscounted All Average PRB (v11)
    * Score: TBD
    * Discount: 0.99 -> 1
 * Undiscounted All Average PRB (v12)
    * Score: TBD
    * Discount: 0.99 -> 1



 * "Tuned" Full-Buffer Minimum Error Threshold
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
