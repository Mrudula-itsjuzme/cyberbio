# Final Attack Strategy Comparison

## 1. Motivation

This comparison asks what happens when fundamentally different adversarial search strategies operate under shared constraints. It does not try to make every method win and does not modify the frozen canonical pipeline.

## 2. Research Questions

Twenty questions and evidence-qualified answers are maintained in `docs/RESEARCH_QUESTIONS.md`. The central result is that no family demonstrates a reproducible, fair-cost, cross-domain advantage sufficient for canonical promotion.

## 3. Shared Comparison Protocol

Executed classical methods use frozen N=30 source manifests, target checkpoint, validator, seed 42, and target-query budgets 5/20/50. Target queries are capped at Q and proposals at 10Q. Generator/attribution/LLM costs are separate. Missing evidence is `NA`.

## 4. Domains

Bio-Cyber is a fixed-length A/C/G/T classification setting with label-flip success. Materials is polymer-SMILES regression with continuous prediction drift and RDKit/canonicalization/tokenization validity. Their success and validity semantics must not be pooled.

## 5. Attack Families

EXECUTED RESULT means a provenance-bearing real run. EXPLORATORY METHOD means implemented but not promotion-ready. BLOCKED METHOD has no efficacy measurement. REJECTED METHOD was considered and intentionally not implemented.

## 6. Random Search

Random valid edits are the baseline. They are cheap, transparent, and necessary to show whether adaptive machinery adds value.

## 7. MCMC

MCMC uses the implemented simplified Metropolis rule over absolute drift. It can exploit locally improving states but tight validity neighborhoods and mixing limit its advantage. Results are exploratory, not a universal win.

## 8. Evolutionary Search

Population mutation, selection, and elitism broaden exploration. It is descriptively strong in some Materials budget-50 comparisons, but proposal cost, N=30, and a single seed prevent promotion.

## 9. Attribution-Guided Search

Occlusion selects high, low, or random positions. Low/random are controls. Counting one attribution query per token shows that high attribution is not consistently superior end-to-end.

## 10. Generative Attacks

The learned proposal model is a GRU denoising sequence autoencoder trained by masked cross entropy. Bio trained and untrained runs exist. The trained Materials run accepted zero proposals. The true Materials untrained control did not complete; a hardcoded substitute was removed. Therefore no learned-validity or cross-domain generator advantage is supported.

## 11. LLM Attacks

BLOCKED METHOD. The proposal-only interface, prompts, accounting, and validator handoff exist, but recorded sweeps used a mock provider. Without real credentials/provider execution, there are no LLM efficacy results.

## 12. RL Attacks

BLOCKED METHOD. An MDP protocol is specified, but no real environment, reward coefficients, trained policy, or held-out evaluation exists. Archived mock code is excluded.

## 13. GAN Feasibility

REJECTED METHOD. Discrete sampling, invalid outputs, unstable adversarial training, and mode collapse make a GAN poorly justified here. The existing generator is not a GAN.

## 14. Mathematical Formulation

All objectives, constraints, transitions, stopping rules, and cost definitions appear in `docs/ATTACK_MATH.md`.

## 15. Validity Constraints

Bio validity is alphabet/length. Materials validity is RDKit parse, canonicalization, vocabulary compatibility, and implemented constraints. These do not prove chemical synthesizability or physical property preservation.

## 16. Query Accounting

The final table separates target queries, proposals, attribution queries, LLM calls/tokens, and runtime. Generator inference is retained in source artifacts. Attribution is not free; invalid proposals do not consume target queries but do consume proposal budget.

## 17. Cross-Domain Results

Classical methods yield measurable drift in both domains, with rankings changing by budget/domain. The generator does not generalize successfully to Materials validity. No LLM/RL/GAN cross-domain conclusion exists.

## 18. Statistical Results

Two-sided paired Wilcoxon tests compare each classical method with Random on matched sources/budgets. Results and median paired differences are in `results/final/statistical_results.json`. They are exploratory and unadjusted for multiple comparisons; no blanket “statistically better” claim is made.

## 19. Efficiency Analysis

Drift per target query alone favors methods that offload work. Therefore proposal count, attribution cost, generator training/inference, runtime, and blocked API/compute costs must accompany it. See the final table and figures.

## 20. Negative Results

Attribution is not a consistent win; MCMC has no universal advantage; Materials generator validity is zero in the trained run; the Materials untrained ablation is incomplete; real LLM and RL efficacy is unknown. Full list: `docs/NEGATIVE_RESULTS.md`.

## 21. Failure Modes

Validity starvation, proposal-cap termination, local-chain mixing, population collapse at small budgets, noncausal attribution, autoregressive invalidity, mock-provider contamination, reward sparsity, and domain-incomparable success definitions are material risks.

## 22. Why Some Methods Failed

Chemical syntax sharply constrains token edits and generation. Attribution points to sensitive positions, not necessarily editable valid ones. Learned decoding used here did not produce accepted Materials proposals.

## 23. Why Some Methods Were Not Run

LLM lacked real external credentials/provider evidence. RL lacked implementation and compute. GAN was rejected on methodological fit. DFT was excluded from the inner loop on cost and availability grounds.

## 24. Threats to Validity

N=30, seed 42, exploratory multiplicity, hardware-dependent timing, different domain semantics, limited chemical oracle, checkpoint/tokenization coupling, fixed attribution, and absence of external physical validation limit inference.

## 25. What Should / Should Not Enter the Canonical Framework

Keep Random as a baseline and the explicit accounting/validator contracts as engineering practice. Do not promote MCMC, Evolutionary, Attribution, Generator, LLM, RL, or GAN as a canonical attack until multiseed evidence shows a meaningful fair-cost advantage with valid outputs.

## 26. Limitations

The study is a branch-level comparison, not a claim of physical molecular adversariality or clinical/biological danger. It lacks a completed Materials generator control, real LLM execution, RL execution, and physical oracle.

## 27. Future Work

Complete the real Materials untrained control with per-proposal timeouts, pin and run a real LLM provider, implement RL only with preregistered reward and held-out sources, repeat classical comparisons across seeds, and use an independent physical oracle on a small selected set.

## 28. Conclusion

The robust conclusion is methodological: adaptive searches expose different model behaviors and costs, but no strategy currently clears the bar for canonical promotion. Failures, blockers, and rejected ideas are part of the result.
