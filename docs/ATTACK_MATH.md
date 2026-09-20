# Attack Mathematics

## Common notation and constraint set

For target model `f`, clean input `x`, and validator-defined feasible set `C(x)`, every executed search solves the black-box objective

`x_adv* = argmax[x' in C(x)] |f(x') - f(x)|`.

Bio-Cyber additionally defines success as a label flip. Materials deliberately has no binary success threshold; drift is continuous. `C(x)` is the A/C/G/T alphabet and fixed length for Bio-Cyber. For Materials it requires RDKit parsing, canonicalization, historical-vocabulary tokenization, and the constraints implemented by the benchmark. RDKit validity is not physical plausibility. Representation-preserving attacks may state `semantic(x_adv) = semantic(x)` only when canonical equivalence establishes that fact. Chemistry-changing edits make no claim that the physical target is unchanged.

The effective target-query budget is `Q`: `q_attack <= Q`. Candidate-generation attempts are capped at `10Q` in the cross-domain benchmark. Attribution cost is separate: `q_total = q_attack + q_attr`, with `q_attr = token_count(x)` for the recorded occlusion method. Efficiency reported here is `mean absolute drift / mean target-model queries`; proposal, attribution, generator, and LLM costs remain separate columns.

## Random / rule-based

The proposal kernel samples a permitted edit and retains the best valid candidate. Its score is `J(x')=|f(x')-f(x)|`; it stops at `Q` target queries or `10Q` proposals. It is the minimum-assumption baseline.

## MCMC

State is the current valid sequence. Given proposal kernel `q(x'|x)`, general Metropolis-Hastings accepts with

`alpha = min(1, pi(x') q(x|x') / (pi(x) q(x'|x)))`.

The recorded cross-domain implementation uses the simplified symmetric rule `alpha=min(1, exp((J(x')-J(x))/10))`; thus `pi(x) proportional exp(J(x)/10)` and the `q` ratio is assumed to cancel. Invalid proposals never query the model. Stopping is `Q` target queries or `10Q` proposals. This local chain can retain improving states but can mix poorly under tight chemical constraints.

## Evolutionary search

The candidate pool uses mutation, tournament-style selection, elitism, and fitness `F(x')=|f(x')-f(x)|`. No extra penalty is present in the recorded cross-domain implementation because invalid candidates are filtered before scoring. Search stops at `Q`, proposal cap, or patience. Global population exploration costs more proposals and can degenerate at small budgets.

## Attribution-guided search

Occlusion attribution for token `i` is `a_i=|f(x)-f(mask_i(x))|`. High, low, and random select positions by maximum, minimum, and random attribution respectively; each uses the same drift score. `q_attr=len(tokenize(x))`, so `q_total=q_attack+q_attr`. Low and random are necessary controls: attribution can be unstable, noncausal, or point to positions with few valid edits.

## Denoising generator

The implemented learned proposer is a bidirectional-GRU encoder plus GRU decoder, not a GAN. It is trained with teacher-forced masked cross entropy:

`L_gen(theta) = - sum[t: y_t != PAD] log p_theta(y_t | y_<t, corrupt(x))`.

At inference it samples autoregressively after source corruption, validates candidates, and scores only accepted candidates by `J`. Generator calls and proposals do not consume target queries but are logged. The trained Materials run produced no accepted proposals; the true untrained Materials control did not complete and is not evidence.

## LLM proposal attacker

An LLM would sample `x' ~ p_LLM(.|prompt(x,constraints,feedback))`. The authoritative objective and validator remain `J` and `C(x)` above; the LLM cannot validate itself. Costs are `(API calls, input tokens, output tokens, latency, target queries)`. Only mock-provider development tests exist, so no measured objective values enter the final table.

## RL and GAN-like approaches

The proposed RL MDP has state `s_t=(x,x_t,f(x),f(x_t),budget_t,validity history)`, edit action `a_t`, transition through edit plus validator, and proposed reward `r_t=lambda_d*Delta J-lambda_i*I[invalid]-lambda_e*edit_cost-lambda_q*query_cost`. No coefficients or trained policy exist; this is a protocol, not an executed result.

A GAN would optimize `min_G max_D E_data log D(x)+E_z log(1-D(G(z)))`, but discrete sampling blocks ordinary backpropagation and introduces mode-collapse and invalid-string risks. No GAN was implemented.
