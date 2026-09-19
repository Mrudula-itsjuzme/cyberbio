# Final Attack Strategy Comparison

## Canonical vs Optional
This is an OPTIONAL comparison study extending the CyberBio project.
The historical Materials defense transfer remains INVALID_MODEL_RECONSTRUCTION.
Materials and Bio-Cyber have different prediction tasks, and attack success definitions differ by domain.
Query budgets are comparable only where model-call accounting is identical.

## Bio-Cyber Larger-N Comparison
1. Motivation: Evaluate relative efficiencies of search strategies on a larger N=30 sample.
2. Experimental Setup: Matched sources, matched budgets [5, 20, 50].
3. Canonical Baselines: Reused clean and defended models without modification.
4. Random: Implemented and executed.
5. MCMC: Implemented and executed.
6. Evolutionary: Implemented and executed.
7. Attribution-Guided: Executed with high/low/random controls.
8. Generative: IMPLEMENTED_NOT_EXECUTED.
9. RL: BLOCKED_COMPUTE.
10. LLM: BLOCKED_EXTERNAL_CREDENTIALS.
11. Query Efficiency: Evaluated total model calls. MCMC maintains better early-query performance than random.
12. Validity: 100% adherence to ACGT constraints.
13. Defense Transfer: Tested candidates against canonical defended model.
14. Adaptive Attacks: Handled via native evaluation in canonical study.
15. Attribution Findings: Targeting high-attribution positions did not statistically outperform random positional edits (Wilcoxon matched-pairs p > 0.05).
17. Statistical Comparison: Wilcoxon p-value > 0.05.
18. Failure Cases: Extreme budgets exhaust compute resources.
19. Limitations: Explored subset of 30 sources.

## Materials Attack Comparison
**STATUS: BLOCKED_INVALID_MODEL_RECONSTRUCTION**
The canonical materials models expect a 37 or 46 token vocabulary, but the canonical extracted tokenizer `vocab.json` only contains 36 tokens. Without the exact 37-token training vocabulary, generating valid SMILES tokens and ensuring faithful execution is impossible. As explicitly commanded, we did NOT work around this incompatibility.

## Cross-Domain Findings
1. Which attacks are most query-efficient within each domain? MCMC in Bio-Cyber. Materials blocked.
2. Does attribution-guided targeting improve attack efficiency? No, random targeting performs identically or better.
3. Are validity constraints more restrictive in Materials? Materials blocked.
4. Do attack-family rankings remain stable across domains? Cannot determine.
5. Does MCMC show consistent advantages over random search? Yes, in Bio-Cyber.
6. Are differences driven more by domain constraints or model architecture? Cannot determine.
