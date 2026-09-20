# Verification of Thompson Sampling Search

## 1. Mathematical Formulation
The adaptive attacker portfolio treats the selection of the perturbation operator (e.g., Substitution, Deletion, Insertion, Rearrangement) as a Multi-Armed Bandit problem. We implemented a Beta-Bernoulli Thompson Sampler.

For each arm $k$:
- Prior: $\theta_k \sim \text{Beta}(\alpha_k, \beta_k)$, initialized to $\alpha=1, \beta=1$.
- Sampling: At iteration $t$, sample $\hat{\theta}_{k,t} \sim \text{Beta}(\alpha_{k,t-1}, \beta_{k,t-1})$.
- Action: Select arm $k^* = \arg\max_k \hat{\theta}_{k,t}$.
- Reward: Observe $r_t \in \{0, 1\}$.
- Update: If the candidate is a fresh proposal, $\alpha_{k,t} = \alpha_{k,t-1} + r_t$, $\beta_{k,t} = \beta_{k,t-1} + (1 - r_t)$. Duplicates skip the update.

## 2. Exact Reward Definition
The reward mechanism handles model queries, chemical validity, and duplicate caching:
- **`binary` (default)**: $r = 1.0$ if the candidate is chemically valid and its prediction drift exceeds the `threshold` (baseline MAE). $r=0.0$ otherwise.
- **`validity-aware binary`**: $r=1.0$ if valid (regardless of drift), else $r=0.0$.
- **`continuous`**: $r = \min(1.0, \max(0.0, \text{drift} / N))$ for valid candidates, else $r=0.0$.

## 3. Query Accounting & Rejection
- We explicitly split resources into `target_query_budget` (model inference) and `proposal_budget` (generation attempts, $k \times \text{query\_budget}$).
- If an arm generates an invalid candidate, it is rejected by the validator before querying the surrogate model.
- The target query budget is **not** consumed, but the arm receives a reward of $r=0.0$ and its posterior is updated ($\beta \leftarrow \beta + 1$).
- Cached/duplicate candidates do not consume target queries and do **not** update the posterior, preventing artificial confidence inflation.

## 4. Test Results and Sanity Experiment
We ran a sanity experiment on 5 fixed test sources over budgets of 10 and 25 queries, across 3 random seeds. The controller was compared against a Uniform random portfolio and independent static searches.

**Observations:**
1. **Source Dependency**: It is **SUPPORTED** that different sources favor different edits. `row_3581` consistently preferred Substitution across seeds, while `row_3577` favored Insertion.
2. **Exploitation**: It is **OBSERVED** that Thompson rapidly learns to exploit a successful arm once a high-reward candidate is found.
3. **Cumulative Reward / Regret**: Our empirical pseudo-regret analysis tracking $T \times \mu_{best} - \sum r_t$ shows **SUGGESTIVE** evidence that Thompson minimizes regret better than Uniform selection in the later stages of the budget, though early-budget performance is largely bound by uniform-like exploration.

## 5. Failure Modes
- **Near-Zero Margins**: In some sources (e.g., `row_3578`), the margin between the best and second-best arm is minimal. Thompson may flutter between arms in these regimes, offering no strict advantage over Uniform.

## 6. Decision Gate & Conclusion
**Classification**: `WORKING_BASELINE` & `ADAPTIVE_SIGNAL_FOUND`.

**Contextual Thompson Justification**: Because the hindsight best arm strictly varies based on the source sequence, a non-contextual Thompson sampler wastes queries exploring. Contextual Thompson is therefore recommended as the next architectural step.
