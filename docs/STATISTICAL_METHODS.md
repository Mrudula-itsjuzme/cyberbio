# Statistical Methods

Paired comparisons use the same source IDs and target-query budgets. For each non-random classical condition, a two-sided Wilcoxon signed-rank test compares per-source absolute drift with Random. The null is that the paired-difference distribution is symmetric about zero. We report `n`, statistic, exact library p-value, and median paired drift difference as a robust effect-size summary in `results/final/statistical_results.json`.

Wilcoxon was chosen because N=30 is small and drift need not be normally distributed. Paired t-tests were not added without normality justification. Tests are exploratory: many domain/budget/method comparisons are present, and no family-wise correction was preregistered. Consequently, unadjusted p-values cannot support broad “statistically better” claims. Confidence intervals already recorded in legacy aggregation remain descriptive; the final table emphasizes effect magnitude and query cost.

Generator trained/untrained inference is not run for Materials because the real control is incomplete. LLM, RL, and GAN have no measured sample and receive no significance test. Missing values are `NA`, never zero-filled. Runtime is hardware-dependent and is summarized, not inferentially tested.
