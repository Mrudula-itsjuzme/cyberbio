# Thompson Sampling Reward Semantics

This document defines the formal reward functions used to update the Beta-Bernoulli posteriors in the `ThompsonSamplingSearch` algorithm.

## Motivation
In adversarial sequence generation, a typical search algorithm aims to maximize a continuous value (e.g. prediction drift). However, classical Thompson Sampling for Beta-Bernoulli bandits requires the reward $r \in \{0, 1\}$. 
Additionally, adversarial candidates must pass a chemical validity checker (e.g., proper valency). We must carefully define how valid and invalid candidates update the search posterior.

## Supported Reward Modes

### 1. `binary`
This mode strictly implements a standard Beta-Bernoulli setup, treating the problem as finding a candidate that exceeds a predefined drift threshold.

**Definition:**
- $r = 1.0$ if candidate is valid AND drift $\ge$ threshold
- $r = 0.0$ otherwise

**Behavior:** Arms that frequently produce chemically valid, high-drift candidates receive a positive update ($\alpha \leftarrow \alpha + 1$). All other scenarios (invalid syntax, valid but low drift) count as failure ($\beta \leftarrow \beta + 1$).

### 2. `validity-aware binary`
This mode separates the goal of finding chemically valid space from the goal of maximizing drift. In this regime, the bandit learns purely which structural operations preserve chemical syntax.

**Definition:**
- $r = 1.0$ if candidate is valid
- $r = 0.0$ if candidate is invalid

**Behavior:** High-drift is ignored for the posterior update. This mode is useful if certain edit families (e.g., long insertions) consistently fail chemical validation and should be abandoned early. (Note: The underlying search mechanism still greedily adopts candidates with higher drift, but the *bandit* only tracks validity).

### 3. `continuous`
This mode implements a pseudo-Bernoulli heuristic update. It leverages the magnitude of the drift to provide partial credit. While not theoretically exact for a standard Beta-Bernoulli conjugate, it empirically guides the search toward higher-yield arms more smoothly than a hard binary threshold.

**Definition:**
Let $N = \max(\text{threshold}, 10^{-5})$ be a normalization factor.
- $r = \min(1.0, \max(0.0, \frac{\text{drift}}{N}))$ if candidate is valid
- $r = 0.0$ if candidate is invalid

**Behavior:** Valid candidates that produce partial drift receive partial reward (e.g., 0.5), leading to fractional updates to $\alpha$ and $\beta$. Invalid candidates still strictly fail.

## Implementation Details
The reward mode can be selected by passing `reward_mode` to the `ThompsonSamplingSearch` constructor. The update logic happens precisely after the surrogate model query, ignoring subsequent rejection reasons (like duplicate caches) to ensure the arm is penalized or rewarded strictly for its generative output quality.
