# Negative and Null Results

- Attribution-high did not establish a universal advantage over random, and attribution overhead worsens end-to-end efficiency.
- MCMC's descriptive gains do not constitute a general statistically supported win under all budgets/domains.
- Attribution-low often starved under Materials validity constraints; proposal-cap termination is evidence, not missing data.
- The trained Materials denoising generator recorded zero valid/accepted proposals at budgets 5, 20, and 50. It therefore does not demonstrate learned chemical validity.
- The Materials untrained control never completed as a real run. The former fast script hardcoded zero drift, 200 proposals, and 15 seconds; its CSV was removed and summary status is `incomplete`.
- The claim that zero untrained validity “conclusively proves” learned weights helped was retracted: the number was fabricated and the trained run was also zero-valid.
- LLM development runs used `MockLLMProvider`; real efficacy is `BLOCKED_EXTERNAL_CREDENTIALS`.
- RL has no trained policy or real result and is `BLOCKED_COMPUTE`.
- GAN-like search was not implemented and is `REJECTED_METHOD`, not a negative measured result.
- Materials has no canonical binary success threshold, so success rate stays `NA` rather than being manufactured.
- Cross-domain validity and success are not numerically equivalent; a pooled universal-winner claim is unsupported.
