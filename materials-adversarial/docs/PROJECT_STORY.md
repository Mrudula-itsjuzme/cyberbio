# Project Story: Evaluating Adversarial Robustness in Polymer Property Prediction

The journey of this repository traces a systematic evolution from identifying representation vulnerabilities to confronting the physical reality of chemical modifications.

## The Sequence Vulnerability
The project began by examining sequence-based predictors that operate on SMILES string representations of polymer repeat units. Initial experiments exposed a severe vulnerability: the canonical Transformer baseline exhibited an equivalent-SMILES drift of `0.614 eV`. Merely serializing the identical 2D chemical graph from a different starting atom caused the model's bandgap prediction to fluctuate wildly, exposing a failure of the model to learn the underlying chemical invariant.

## Attempted Defenses and Semantic Hypothesis
We hypothesized that the model lacked a "semantic branch" to differentiate syntax (SMILES ordering) from semantics (chemistry). This two-branch hypothesis was rejected when adversarial fine-tuning resulted in post-hoc collapse, indicating that the representation itself was structurally flawed for robustness.

## The Architecture Solution: GraphMPNN
To resolve the representation vulnerability structurally, we pivoted to a Message Passing Neural Network (GraphMPNN). By operating directly on the molecular graph rather than a string serialization, equivalent-SMILES drift was reduced to effectively `0 eV` (limited only by numerical precision). The GraphMPNN also achieved a strong clean validation MAE of `0.411 eV`.

## Chemistry-Changing Sensitivities
While the representation vulnerability was solved, the GraphMPNN remained highly sensitive to valid, bounded chemical modifications (substitutions and deletions). The repaired bounded adaptive multi-substitution search produced a maximum observed GraphMPNN prediction drift of approximately `3.19 eV`. (Deletion was evaluated separately as a fixed stress).
*(Note: An earlier search bug in Phase 12 falsely reported 5.961 eV due to edit creep (states not strictly constrained to <=3 edits from the original source); this bug was caught in Phase 12B, corrected, and the 5.961 eV result was marked invalid).*

## The Oracle Imperative
Finding a 3.19 eV prediction change raised a fundamental question: Is the model wrong, or does the chemical edit *actually* change the physical bandgap by 3.19 eV? Without an independent physical reference, this drift cannot be definitively classified as an adversarial error. It might simply be accurate physics.

## Surrogate Structure and the External Block
To resolve this, we required a physical oracle (Quantum ESPRESSO). However, we discovered that the source dataset's exact DFT protocol and 3D periodic geometries were incomplete or missing. We designed a transparent, reproducible `CALIBRATABLE_SURROGATE` protocol that constructs finite capped oligomers (n=2, 3, 4) using ETKDG and MMFF94. A 9-job calibration pilot was strictly prepared, hashed, and bundled for cluster handoff.

Ultimately, the real calibration execution was blocked because local access to the required HPC infrastructure (SLURM, `pw.x`, pseudopotentials) is unavailable. The project branch is successfully frozen at this external compute boundary, awaiting physical validation to complete the attack→defend loop.
