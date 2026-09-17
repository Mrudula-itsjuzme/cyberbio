# Future Work Roadmap

## Near Term
* **Execute 9-Job Surrogate QC Pilot**: Transfer the frozen `oracle_surrogate_qc_bundle` to a real HPC SLURM cluster and run the protocol execution script.
* **Assess Length Sensitivity**: Analyze the parsed results for n=2, n=3, and n=4 chain lengths to evaluate surrogate predictability.
* **Expand Calibration Scope**: If the pilot's trend is mathematically interpretable (and technically stable), expand the pilot to the remaining 3 benchmark polymers.
* **Evaluate Adversarial Pairs**: Once calibrated, test 5 adversarial source/candidate polymer pairs against the oracle to determine physical truth.

## Mid Term
* **Oracle-Backed Defender Training**: Use ground-truth oracle evaluations as the objective function for robust adversarial training.
* **Attack→Defend→Re-attack Loop**: Close the adversarial loop iteratively, preventing model drift from bypassing prior defenses.
* **Uncertainty/Calibration Analysis**: Incorporate explicit predictive uncertainty limits to reject adversarial perturbations rather than blindly accepting or rejecting them.
* **Richer Chemistry Edit Operators**: Expand the bounds of validity beyond single-atom substitutions and deletions to encompass topological rearrangements.
* **Retrieval Manipulation**: Study dataset poisoning and retrieval-augmented manipulation in materials prediction.

## Long Term
* **LLM Proposal Attacker + Strict Validator**: Introduce generative language models as intelligent combinatorial search agents, constrained by deterministic validity checkers.
* **Generative Attacker**: Direct sequence-to-sequence or graph-to-graph generative models designed explicitly to maximize loss.
* **GAN/Minimax Experiments**: True dual-model adversarial settings for continuous property fields.
* **Protein/DNA Domain Adapters**: Swap the representation adapter to apply the generalized framework to genomic and proteomic sequences.
* **Attacker/Defender Dynamics**: Large-scale, honeypot-style simulation modeling open-ended adversarial arms races in scientific computing.
