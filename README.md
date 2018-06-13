# retro-agents

1. [Retro Contest](https://contest.openai.com/)
2. [Retro Contest Writeup](/results/writeup/iclr2018_conference.pdf)
3. [Retro Contest Retrospective Blog Post](https://www.endtoend.ai/blog/i-learned-dqns-with-openai-competition)



## Overview

This is a repository containing agents used for [OpenAI's Retro Contest](https://contest.openai.com/). The competition was to create the best agent for Sonic the Hedgehog games using only 1 million timesteps and 12 hours of time on a VM with 6 E5-2690v3 cores, 56GB of RAM, and a single K80 GPU.

Below is a video of my agent on a custom stage that finished training. You can see that the agent found a glitch in the stage and used it to its favor.

![Video of my trained agent on a custom stage](video.gif)



## Ideas

For a more detailed discussion of ideas used, please check the [writeup](/results/writeup/iclr2018_conference.pdf).


### Tested Ideas

These are ideas that have been implemented and tested. You can read the results [here](/results/RESULTS.md).

#### Sampling

 * Dual Sampling: Some uniformly, Some with priority

#### Collection

 * Threshold with Buffer TD-error Average
 * Threshold with Overall TD-error Average
 * Threshold with Average of Buffer and Overall TD-error Average
 * Threshold with Buffer TD-error Exponential Average
 * Stochastic Threshold with Buffer TD-error Average
 * Threshold with Minimum TD-error in Buffer
 * Stochastic Threshold with Maximum TD-error
 * Stochastic Threshold with (Maximum - Minimum) TD-error

#### Deletion

 * Minimum TD-error Deletion
 * Stochastic TD-error Deletion
 * TD-error Delta Deletion
 * Stochastic Deletion with TD-error deltas

#### Miscellaneous

 * Rainbow DQN without Dueling
 * Double Replay Memory: 1 Short Term (Episodic) Replay Memory, 1 Long Term (Lasting) Replay Memory
 * No Minimum Buffer Size


### Untested Ideas

These are ideas I could not test because there was not enough time to implement.

#### Sampling

 * Double Sampling: Two filters for sampling
 * URB variants of tested ideas (for comparison)
 * Variants of TD-error prioritization shown in *Prioritized Experience Replay* paper

#### Collection

 * Threshold with Decaying Buffer TD-error Average
 * Overall TD-error Exponential Average
 * Non TD-error prioritization shown in *Prioritized Experience Replay* paper

#### Deletion

 * Uniform Deletion
 * Staleness penalty for Deletion
 * No deletion: Flush Replay Buffer periodically
 * Sigmoid Stochastic Deletion with TD-error deltas

#### Miscellaneous

 * Encode states and compute similarities between states in replay memory
 * Double Replay Memory: 2 Long Term Replay Memory
 * Using JERK to fill replay memory of Rainbow DQN
 * Periodic Recalculation of all errors in replay memory
 * Hyperparameter Tuning
 * Vary hyperparameters for parts of training phase
 * Cyclic NoisyNet: Add additional noise periodically
 * Subtract time from reward in early training phase to penalize wasting timesteps
