# Research Paths Explored

| Method / idea | Why / question | Status | What was learned; keep/drop; claim impact |
|---|---|---|---|
| Random valid edits | Minimum-assumption baseline | EXECUTED | Essential comparator; retained; supports baseline-only findings. |
| Substitution / insertion / deletion | Test local edit operators | PARTIALLY_EXECUTED | Implementations differ by domain; do not merge operator effects; retained as search primitives. |
| MCMC | Does stateful local search beat random? | EXECUTED | Often higher descriptive drift, but significance and constraint effects vary; exploratory candidate only. |
| Evolutionary | Does a population explore globally? | EXECUTED | Strong descriptive Materials drift at budget 50, but expensive and single-seed; not promotable. |
| Attribution high / low / random | Does salience improve efficiency? | EXECUTED | High is not uniformly superior after attribution cost; controls expose weak causal interpretation. |
| Sequence-length hypothesis | Is vulnerability a length shortcut? | EXECUTED elsewhere / audited | Association does not establish mechanism; it narrows claims. |
| RandomSMILES | Representation sensitivity without intended chemistry change | EXECUTED in canonical research | Relevant evidence, but distinct from chemistry-changing comparison attacks. |
| Defense transfer | Does one defense cover other attacks? | PARTIALLY_EXECUTED | Bio evidence exists; Materials compatibility was not established; no universal defense claim. |
| Trained generator | Can learned proposals improve validity? | EXECUTED | Bio proposals were valid; Materials recorded zero accepted proposals. No cross-domain win. |
| Untrained generator | Is performance due to weights? | EXECUTED in Bio; BLOCKED_COMPUTE in Materials | Materials fast result was fabricated and removed. Bio control exists; no Materials ablation conclusion. |
| LLM proposal attacker | Can semantic priors propose useful edits? | BLOCKED_EXTERNAL | Mock tests validate plumbing only; credentials/provider execution absent; no efficacy claim. |
| RL policy attacker | Can a policy adapt across steps? | BLOCKED_COMPUTE | No implementation/run; protocol only. |
| GAN idea | Can adversarial training learn proposals? | REJECTED_METHOD | Poor fit for discrete constrained strings; existing generator is not a GAN. |
| Physical oracle / DFT | Do candidates retain physical plausibility? | NOT_RUN | Too costly for the inner loop; RDKit is insufficient; blocks physical claims. |
| Graph model / GNN | Would representation choice remove string sensitivity? | NOT_RUN | A target-model replacement is not a fair attack comparison. Future robustness study. |
| Gradient attacks | Can white-box gradients optimize tokens? | REJECTED_METHOD | Discrete inputs and cross-method black-box parity make direct gradients incomparable. |
| Explainability-guided attacks | Does model attribution guide useful edits? | EXECUTED | Same as attribution controls; attribution overhead must be counted. |
| Leakage checks | Were test labels/data used by proposals? | EXECUTED | Frozen source manifests and training exclusions exist; strengthens provenance, not efficacy. |
| Shortcut investigations | Is drift driven by length/token artifacts? | EXECUTED elsewhere / audited | Warns against mechanistic interpretation and promotion. |

Archived mock RL and invalid completion outputs remain only under `archive/` as research-history evidence and never feed final analysis.
