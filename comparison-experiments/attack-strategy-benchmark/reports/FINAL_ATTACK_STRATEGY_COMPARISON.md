# Final Attack Strategy Comparison

## Canonical vs Optional
This is an OPTIONAL comparison study extending the CyberBio project.
The Materials vocabulary has been successfully recovered (see `MATERIALS_VOCABULARY_FORENSICS.md`), unblocking cross-domain comparison.
Query budgets are comparable only where model-call accounting is identical.

## Bio-Cyber Larger-N Comparison
1. Motivation: Evaluate relative efficiencies of search strategies on a larger N=30 sample.
2. Experimental Setup: Matched sources, matched search budgets [5, 20, 50].
3. Canonical Baselines: Reused clean and defended models without modification.
4. Random: Implemented and executed.
5. MCMC: Implemented and executed.
6. Evolutionary: Implemented and executed.
7. Attribution-Guided: Executed with high/low/random controls.
8. Generative: IMPLEMENTED_NOT_EXECUTED.
9. RL: BLOCKED_COMPUTE.
10. LLM: BLOCKED_EXTERNAL_CREDENTIALS.
11. Query Efficiency: Evaluated total model calls. Accounting now separates `search_queries` and `attribution_queries`.
12. Validity: 100% adherence to ACGT constraints.
13. Defense Transfer: Tested candidates against canonical defended model.
14. Adaptive Attacks: Handled via native evaluation in canonical study.
15. Attribution Findings: Targeting high-attribution positions using model-occlusion sensitivity does not outperform random positional edits under matched budget conditions.
16. Query Accounting: Strict. Model calls increment exactly per inference batch/sequence.
17. Statistical Comparison: Wilcoxon p-value > 0.05 between Attribution-High and Random.
18. Failure Cases: Extreme budgets exhaust compute resources.
19. Limitations: Explored subset of 30 sources.

## Materials Attack Comparison
**STATUS: UNBLOCKED**
The Materials canonical model was proven to natively utilize the 36-token vocabulary file through a 1-indexed ad-hoc mapping layer during training. The 37-token input shape expectation perfectly aligns with this recovered logic (see `MATERIALS_VOCABULARY_FORENSICS.md` for full proof). The Materials benchmark suite can now safely execute.

## Cross-Domain Findings
1. Which attacks are most query-efficient within each domain? MCMC in Bio-Cyber. Materials execution pending.
2. Does attribution-guided targeting improve attack efficiency? No, random targeting performs identically or better.
3. Are validity constraints more restrictive in Materials? Materials execution pending.
4. Do attack-family rankings remain stable across domains? Cannot determine yet.
5. Does MCMC show consistent advantages over random search? Validated in Bio-Cyber only.
6. Are differences driven more by domain constraints or model architecture? Cannot determine yet.
