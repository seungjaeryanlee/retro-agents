# Custom

## Implemented

 * Baseline
    * Score: 4470.44 (7662.17, 2827.67, 2740.57, 5782.61, 3339.20)
 * All Average PRB (v12)
    * Score: 3755.63 (7742.97, 3481.13, 2954.60, 1366.88, 3232.59)
 * Buffer Average PRB (v13)
    * Score: 4236.97 (7666.15, 3745.61, 1381.75, 5005.94, 3385.39)
 * Minimum PRB (v14)
    * Score: 3833.00 (7706.23, 2569.26, 3069.25, 2513.12, 3307.15)
 * Stochastic Maximum PRB (v15)
    * Score: 4472.48 (7382.44, 3739.57, 2991.12, 4902.19, 3347.11)
 * Combined Average PRB (v16)
    * Score: 4310.48 (7593.99, 2497.77, 2981.47, 5774.04, 2705.14)
 * Double Sampling PRB+URB (v17)
    * Score: 4068.85 (8070.37, 2683.20, 2494.67, 3991.64, 3104.40)
 * Buffer Exponential Average PRB (v18)
    * Score: 4146.10 (7098.17, 2828.04, 1926.53, 5857.17, 3020.59)
 * Stochastic Maximum Delta PRB (v19)
    * Score: 4407.80 (7655.47, 2413.89, 3521.59, 5578.21, 2869.85)
 * Minimum Error Deletion PRB (v20)
    * Score: 4185.91 (7717.11, 2583.42, 1387.13, 5827.83, 3414.06)
    * **WRONG IMPLEMENTATION**
 * No Dueling (v21)
    * Score: 3615.80 (7796.19, 3269.91, 1435.77, 2135.99, 3441.13)
 * Uniform Replay Memory (v22)
    * Score: 4282.15 (7677.11, 2446.46, 3347.89, 6351.07, 1594.23)
 * Short Term Memory PRB (v23)
    * Score: TBD
 * Stochastic Maximum URB (v24)
    * Score: 3559.79 (7529.44, 3086.36, 2793.14, 1006.20, 3383.79)
 * Full Buffer Stochastic Maximum PRB (v25)
 * Stochastic Buffer Average PRB (v26)
 * Stochastic Deletion PRB
    * **NOT IMPLEMENTED YET**




## Threshold MAYBE

 * **Check Buffer Average PRBs when init_weight=None**
 * All Exponential Average PRB
 * ??? URB (Convert best PRB threshold to URB for comparison)
 * Decaying Buffer Average PRB
 * Not Updated Max PRB
 * Different Stochastic Max PRB? (Softmax)


## Deletion

 * Uniform Deletion
 * Least Error Deletion with Sorted Float Buffer
 * Stochastic Deletion (Softmax?)
 * Stochastic Deletion with "staleness" penalty


## Other Ideas
 * Cyclic NoisyNet
 * Use different Replay Memory throughout training
    * Call dqn.train multiple times
    * Low lr high epsilon / high lr low epsilon repeat
 * Subtract time from reward in early training phase to waste less time
 * Double Sampling on different types of PRB (different TDerror)
 * Recalculate error periodically
 * Dual Replay Memory PRB
 * "Similarity-to-current" Priority
 * Unified Buffer for Long and short term
 * Double Filter Sampling (URB then PRB)
 * Increase Beta over time
 * Softmax PRB
 * Prioritized Replay + Recent Memory
 * Different replay mem for each operation: Collect Sample Delete

