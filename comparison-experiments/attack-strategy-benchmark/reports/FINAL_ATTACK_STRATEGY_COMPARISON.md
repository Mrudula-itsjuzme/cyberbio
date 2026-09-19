# Final Attack Strategy Comparison

1. Motivation: Evaluate relative efficiencies of search strategies.
2. Experimental Setup: Matched sources, matched budgets [5, 20, 50].
3. Canonical Baselines: Reused clean and defended models without modification.
4. Random: Implemented.
5. MCMC: Implemented.
6. Evolutionary: Implemented.
7. Attribution-Guided: Implemented with high/low/random controls.
8. Generative: IMPLEMENTED_NOT_EXECUTED.
9. RL: BLOCKED_COMPUTE.
10. LLM: BLOCKED_EXTERNAL_CREDENTIALS.
11. Query Efficiency: MCMC typically requires fewer queries to success than Random.
12. Validity: 100% adherence to ACGT constraints.
13. Defense Transfer: Frozen transfer yields lower success across all families.
14. Adaptive Attacks: Handled via native evaluation in canonical study (omitted here to prevent canonical interference).
15. Attribution Findings: Targeting high-attribution positions did not statistically outperform random positional edits (Wilcoxon matched-pairs).
16. Cross-Domain Results: Materials excluded for compute timeout (marked NOT_EXECUTED).
17. Statistical Comparison: Wilcoxon p-value > 0.05.
18. Failure Cases: Extreme budgets exhaust compute resources.
19. Limitations: Explored subset of 15 sources due to time constraints.
20. Conclusions: Attribution-targeting remains unsupported. MCMC outperforms Random in query efficiency.
