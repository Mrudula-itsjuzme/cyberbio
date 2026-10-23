# Research Directions After the Initial Lab

These are hypotheses to test, not novelty claims.

## 1. Feasibility-boundary search

Test whether the largest model drift concentrates near the boundary of the
current validity rules. Compare ordinary validity-SMC with a controller that
allocates a fixed fraction of proposals to high-uncertainty feasibility regions.
This connects the lab to boundary-exploration constrained BO, while the specific
polymer threat model and failure analysis may be application-specific.

## 2. Calibrated validity under distribution shift

Replace the per-family Beta model with a deliberately small contextual model
using source length, edit count, family, token context, and cheap descriptors.
Evaluate calibration and selective risk by source scaffold, not only global
accuracy. The scientific question is whether saved invalid proposals translate
to more effect per target query without shrinking chemical diversity.

## 3. Multi-fidelity robustness search

Use a cheap proxy to rank proposals, the frozen GraphMPNN as the matched-budget
target, and a physical oracle only for a small preregistered subset. Measure
ranking reversals and optimizer's-curse amplification. This is stronger than
optimizing target-model drift alone because it can reveal whether search methods
systematically exploit model error.

## 4. Representation-invariant adversarial search

For each molecular graph, marginalize or stress-test across randomized string
representations. Ask whether a method finds graph changes or merely tokenizer /
serialization sensitivities. Pair graph-equivalent controls with chemistry-
changing candidates and report both populations separately.

## 5. Pareto sampling instead of scalar reward

Treat drift, validity confidence, edit cost, diversity, and uncertainty as a
Pareto problem. Compare scalarized SMC/CEM against feasible Pareto-front methods.
This avoids hiding value judgments in a single penalty weight.

## 6. Diversity-preserving particle search

Measure genealogical collapse, not only terminal unique count. Add stratified or
systematic resampling, repulsive kernels, and family-conditioned rejuvenation.
The core result would be whether preserving multiple modes improves robustness
coverage at a fixed target-query budget.

## 7. Amortization break-even study

Only after repeated-search workloads exist, compare per-instance SMC/CEM with a
small GFlowNet feasibility spike. Include training queries and compute in the
break-even calculation. GFlowNets are established reward-proportional samplers;
the useful question here is when amortization pays for this constrained polymer
search space.

## 8. Search-induced shortcut discovery

Cluster high-drift candidates by edit motif, source scaffold, length change, and
representation change. Then run controlled counterfactual edits to determine
which discovered motifs are causal model shortcuts. This builds directly on the
earlier finding that a length-specific narrative can fail under controls.

## 9. Robustness across checkpoints and model classes

Freeze candidate banks generated against one model, then evaluate them on other
seeds and architectures without adaptive feedback. Separate white-box search
success from transferable failure modes and paired inference.

## 10. Stopping as a statistical decision

Compare fixed query budgets with stopping rules based on posterior probability
of meaningful improvement, ESS collapse, or confidence bounds on best-so-far
gain. Count any calibration/training cost and preregister the stopping rule.

## Primary starting points

- Gardner et al., constrained Bayesian optimization: https://proceedings.mlr.press/v32/gardner14.html
- Tian et al., feasibility-boundary exploration: https://proceedings.mlr.press/v235/tian24g.html
- Eriksson and Poloczek, scalable constrained BO: https://proceedings.mlr.press/v130/eriksson21a.html
- Del Moral, Doucet, and Jasra, SMC samplers: https://www.stats.ox.ac.uk/~doucet/delmoral_doucet_jasra_sequentialmontecarlosamplersJRSSB.pdf
- Bengio et al., GFlowNet foundations: https://www.jmlr.org/papers/v24/22-0364.html
- Shen et al., practical GFlowNet training: https://proceedings.mlr.press/v202/shen23a.html
- Inverse design including polymer stoichiometry and architecture: https://pmc.ncbi.nlm.nih.gov/articles/PMC11650379/
